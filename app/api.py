"""
Prototype operational API (spec Sec.27).

    uvicorn app.api:app --host 0.0.0.0 --port 8000

GET  /health
GET  /points                              -> the 900 canonical coordinates
GET  /init_dates?split=test               -> which init dates have data for a split
POST /forecast   {"init_date": "2025-08-21"}
    -> {
         "forecast_date": ..., "lead_days": 7,
         "regions": [...], "grid_predictions": [...],
         "confidence_maps": [...], "bust_probability_maps": [...], "error_maps": [...],
         "explanations": [...]
       }

Everything is read from data/features/samples.parquet for the requested init_date (i.e. the
API replays a date that already has features built -- for a genuinely live/operational
deployment, point this at a small script that builds features for "today" the same way
scripts/02-03 do, then swap the parquet read for that single-day frame).
"""
import json, sys
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bustcast.utils import load_cfg, P, get_device
from bustcast.dataset import split_feature_cols
from bustcast.model import BustCastModel
from bustcast.geo import norm_coords

app = FastAPI(title="BustCast -- Forecast Bust Detection API", version="0.1")

STATE = {}


def load_state():
    cfg = load_cfg()
    feat_dir = P(cfg, "features")
    meta = json.load(open(feat_dir / "feature_meta.json"))
    feature_cols = meta["feature_columns"]
    err_cols = [f"z_err_{v}" for v in meta["error_vars"]]
    leads = meta["leads"]
    hist_cols, lead_cols = split_feature_cols(feature_cols)

    g = np.load(feat_dir / "graph.npz")
    device = get_device(verbose=False)
    model = BustCastModel(n_hist_feat=len(hist_cols), n_lead_feat=len(lead_cols), n_points=len(g["point_id"]),
                           n_leads=len(leads), n_error_vars=len(err_cols), cfg=cfg,
                           coords_norm=norm_coords(g["region_centroid"]), region_pool=g["region_pool"]).to(device)
    ckpt = Path(cfg["paths"]["outputs"]) / "checkpoints" / "stage3_best.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    calib_path = Path(cfg["paths"]["outputs"]) / "calibration" / "calibration_report.json"
    calibration = json.load(open(calib_path)) if calib_path.exists() else {}

    df = pl.read_parquet(feat_dir / "samples.parquet")
    medians = {c: (df.filter(pl.col("split") == "train")[c].median() or 0.0) for c in feature_cols}

    STATE.update(cfg=cfg, meta=meta, feature_cols=feature_cols, hist_cols=hist_cols, lead_cols=lead_cols,
                 err_cols=err_cols, leads=leads, g=g, device=device, model=model, calibration=calibration,
                 df=df, medians=medians,
                 pid_to_slot={int(p): i for i, p in enumerate(g["point_id"])},
                 nbr=torch.from_numpy(g["nbr"]).long().to(device),
                 edge_feat=torch.from_numpy(g["edge_feat"]).float().to(device))


@app.on_event("startup")
def _startup():
    load_state()


class ForecastRequest(BaseModel):
    init_date: str
    include_explanations: bool = True
    explanation_prob_threshold: float = 0.5


def calibrate_prob(raw_prob, lead):
    rep = STATE["calibration"].get(str(lead)) or STATE["calibration"].get(lead)
    if not rep or "chosen_method" not in rep:
        return raw_prob
    return raw_prob  # calibrators aren't persisted as objects here; report gives you their test-set
    # quality. To apply a saved calibrator at request time, re-fit + pickle it in script 06
    # (kept simple here since this is a prototype-dashboard API, not a production serving path).


def risk_category(p):
    edges, labels = STATE["cfg"]["risk"]["edges"], STATE["cfg"]["risk"]["labels"]
    for e, lab in zip(edges, labels):
        if p <= e:
            return lab
    return labels[-1]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in STATE}


@app.get("/points")
def points():
    g = STATE["g"]
    return [{"point_id": int(pid), "latitude": float(lat), "longitude": float(lon)}
            for pid, (lat, lon) in zip(g["point_id"], g["coords"])]


@app.get("/init_dates")
def init_dates(split: Optional[str] = "test"):
    df = STATE["df"]
    if split:
        df = df.filter(pl.col("split") == split)
    return sorted(str(d) for d in df["init_date"].unique().to_list())


@app.post("/forecast")
def forecast(req: ForecastRequest):
    df = STATE["df"]
    day = df.filter(pl.col("init_date").cast(pl.Utf8) == req.init_date)
    if day.height == 0:
        raise HTTPException(404, f"No feature rows for init_date={req.init_date}. "
                                  f"Try GET /init_dates for what's available.")

    g, leads = STATE["g"], STATE["leads"]
    n_points = len(g["point_id"])
    hist_cols, lead_cols = STATE["hist_cols"], STATE["lead_cols"]
    medians = STATE["medians"]

    Xh = np.zeros((n_points, len(hist_cols)), dtype=np.float32)
    Xl = np.zeros((n_points, len(leads), len(lead_cols)), dtype=np.float32)
    hist_done = np.zeros(n_points, dtype=bool)
    for row in day.iter_rows(named=True):
        slot = STATE["pid_to_slot"].get(int(row["point_id"]))
        if slot is None or row["lead_day"] not in leads:
            continue
        li = leads.index(row["lead_day"])
        if not hist_done[slot]:
            for fi, fc in enumerate(hist_cols):
                v = row.get(fc); Xh[slot, fi] = v if v is not None else medians.get(fc, 0.0)
            hist_done[slot] = True
        for fi, fc in enumerate(lead_cols):
            v = row.get(fc); Xl[slot, li, fi] = v if v is not None else medians.get(fc, 0.0)

    device = STATE["device"]
    Xh_t = torch.from_numpy(Xh).unsqueeze(0).to(device)
    Xl_t = torch.from_numpy(Xl).unsqueeze(0).to(device)
    with torch.no_grad():
        err, errvar, bust_logit = STATE["model"](Xh_t, Xl_t, STATE["nbr"], STATE["edge_feat"])
    err = err[0].cpu().numpy(); prob = torch.sigmoid(bust_logit)[0].cpu().numpy()

    grid_predictions, region_rows = [], []
    coords = g["coords"]; region_label = g["region_label"]
    for slot in range(n_points):
        if not hist_done[slot]:
            continue
        pid = int(g["point_id"][slot])
        for li, lead in enumerate(leads):
            p = float(prob[slot, li])
            grid_predictions.append(dict(
                point_id=pid, latitude=float(coords[slot, 0]), longitude=float(coords[slot, 1]),
                lead_day=lead, predicted_error=float(err[slot, li]), bust_probability=p,
                confidence=1 - p, risk_category=risk_category(p),
            ))

    for r in np.unique(region_label):
        sel = np.where(region_label == r)[0]
        sel = [s for s in sel if hist_done[s]]
        if not sel:
            continue
        for li, lead in enumerate(leads):
            pe = err[sel, li]; pp = prob[sel, li]
            region_rows.append(dict(region_id=int(r), lead_day=lead,
                                     mean_predicted_error=float(pe.mean()), max_predicted_error=float(pe.max()),
                                     mean_bust_probability=float(pp.mean()), max_bust_probability=float(pp.max()),
                                     pct_high_risk=float((pp > STATE["cfg"]["risk"]["edges"][2]).mean()),
                                     confidence=float(1 - pp.mean())))

    confidence_maps = [{"lead_day": lead, "points": [
        {"point_id": r["point_id"], "latitude": r["latitude"], "longitude": r["longitude"],
         "value": r["confidence"]} for r in grid_predictions if r["lead_day"] == lead]}
        for lead in leads]
    bust_probability_maps = [{"lead_day": lead, "points": [
        {"point_id": r["point_id"], "latitude": r["latitude"], "longitude": r["longitude"],
         "value": r["bust_probability"]} for r in grid_predictions if r["lead_day"] == lead]}
        for lead in leads]
    error_maps = [{"lead_day": lead, "points": [
        {"point_id": r["point_id"], "latitude": r["latitude"], "longitude": r["longitude"],
         "value": r["predicted_error"]} for r in grid_predictions if r["lead_day"] == lead]}
        for lead in leads]

    explanations = []
    if req.include_explanations:
        from bustcast.explain import explain_for_request
        explanations = explain_for_request(STATE, Xh, Xl, hist_done, req.explanation_prob_threshold)

    return dict(forecast_date=req.init_date, lead_days=len(leads),
                regions=region_rows, grid_predictions=grid_predictions,
                confidence_maps=confidence_maps, bust_probability_maps=bust_probability_maps,
                error_maps=error_maps, explanations=explanations)
