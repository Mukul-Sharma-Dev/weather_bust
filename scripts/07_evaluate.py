"""
Produces:
  outputs/eval/leadwise_metrics.csv         -- MAE/RMSE/PR-AUC/ROC-AUC/F1/Brier/Recall per lead
  outputs/eval/region_aggregation.csv       -- per (region, lead): mean/max error, mean/max P(bust), confidence
  outputs/eval/error_prone_areas.json       -- clustered high-risk regions per lead
  outputs/eval/maps/lead{d}_{actual_error,pred_error,actual_bust,pred_bust,confidence,diff}.png
      six maps per lead day, for a chosen test-split init_date (the one with the highest
      actual bust rate, so the maps actually show something)

Usage: python scripts/07_evaluate.py --checkpoint outputs/checkpoints/stage3_best.pt
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import polars as pl
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, recall_score,
                              mean_absolute_error, mean_squared_error, brier_score_loss)
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bustcast.utils import load_cfg, P, ensure, save_json, get_device
from bustcast.dataset import build_datasets, collate, split_feature_cols
from bustcast.model import BustCastModel
from bustcast.geo import norm_coords
from torch.utils.data import DataLoader


def run_inference(model, loader, nbr, edge_feat, device):
    E, EV, B, Y, M, DATES = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            b2 = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            err, errvar, bust_logit = model(b2["Xh"], b2["Xl"], nbr, edge_feat)
            E.append(err.cpu().numpy()); B.append(torch.sigmoid(bust_logit).cpu().numpy())
            Y.append(batch["Ycomp"].numpy()); M.append(batch["mask"].numpy())
            DATES += batch["init_date"]
    return (np.concatenate(E, 0), np.concatenate(B, 0), np.concatenate(Y, 0),
            np.concatenate(M, 0), DATES)


def leadwise_metrics(pred_err, pred_prob, true_err, true_bust, mask, leads):
    rows = []
    for li, lead in enumerate(leads):
        m = mask[:, :, li].astype(bool)
        pe, te = pred_err[:, :, li][m], true_err[:, :, li][m]
        pp, tb = pred_prob[:, :, li][m], true_bust[:, :, li][m]
        row = {"lead_day": lead, "n": int(m.sum())}
        if len(te):
            row["MAE"] = float(mean_absolute_error(te, pe))
            row["RMSE"] = float(np.sqrt(mean_squared_error(te, pe)))
            row["Pearson_r"] = float(pearsonr(te, pe)[0]) if len(te) > 1 else None
            row["Spearman_r"] = float(spearmanr(te, pe)[0]) if len(te) > 1 else None
        if len(tb) and len(np.unique(tb)) > 1:
            pred_bin = (pp > 0.5).astype(int)
            row["ROC_AUC"] = float(roc_auc_score(tb, pp))
            row["PR_AUC"] = float(average_precision_score(tb, pp))
            row["F1"] = float(f1_score(tb, pred_bin, zero_division=0))
            row["Recall"] = float(recall_score(tb, pred_bin, zero_division=0))
            row["Brier"] = float(brier_score_loss(tb, pp))
            fp = ((pred_bin == 1) & (tb == 0)).sum(); neg = (tb == 0).sum()
            fn = ((pred_bin == 0) & (tb == 1)).sum(); pos = (tb == 1).sum()
            row["False_Alarm_Rate"] = float(fp / neg) if neg else None
            row["Miss_Rate"] = float(fn / pos) if pos else None
        rows.append(row)
    return rows


def make_maps(coords, point_ids, pid_order, true_err, pred_err, true_bust, pred_prob, mask,
              leads, init_dates, out_dir):
    ensure(out_dir)
    bust_rate_by_sample = (true_bust * mask).sum(axis=(1, 2))
    best = int(np.argmax(bust_rate_by_sample))
    d = init_dates[best]
    print(f"Generating spatial maps for the highest-bust-rate test init_date: {d}")
    lat, lon = coords[:, 0], coords[:, 1]
    for li, lead in enumerate(leads):
        m = mask[best, :, li].astype(bool)
        panels = [
            ("actual_error", true_err[best, :, li], "magma"),
            ("pred_error", pred_err[best, :, li], "magma"),
            ("actual_bust", true_bust[best, :, li], "Reds"),
            ("pred_bust_probability", pred_prob[best, :, li], "Reds"),
            ("confidence", 1 - pred_prob[best, :, li], "Greens"),
            ("pred_minus_actual_error", pred_err[best, :, li] - true_err[best, :, li], "coolwarm"),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        for ax, (name, vals, cmap) in zip(axes.flat, panels):
            sc = ax.scatter(lon[m], lat[m], c=vals[m], cmap=cmap, s=18)
            ax.set_title(name); ax.set_xlabel("lon"); ax.set_ylabel("lat")
            plt.colorbar(sc, ax=ax, shrink=0.8)
        fig.suptitle(f"Lead Day {lead} -- init {d}")
        fig.tight_layout()
        fig.savefig(out_dir / f"lead{lead}_maps.png", dpi=110)
        plt.close(fig)


def region_aggregation(region_label, region_names, true_err, pred_err, pred_prob, mask, leads, out_csv):
    rows = []
    for r in np.unique(region_label):
        sel = region_label == r
        for li, lead in enumerate(leads):
            m = mask[:, sel, li].astype(bool)
            if m.sum() == 0:
                continue
            pe = pred_err[:, sel, li][m]; pp = pred_prob[:, sel, li][m]
            rows.append(dict(region=int(r), lead_day=lead, mean_pred_error=float(pe.mean()),
                              max_pred_error=float(pe.max()), mean_bust_prob=float(pp.mean()),
                              max_bust_prob=float(pp.max()), pct_high_risk=float((pp > 0.5).mean()),
                              mean_confidence=float(1 - pp.mean())))
    pl.DataFrame(rows).write_csv(out_csv)


def error_prone_areas(coords, region_label, pred_prob, mask, leads, threshold, min_cells, link_km):
    from scipy.spatial import cKDTree
    from bustcast.geo import haversine_km
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    areas = {}
    for li, lead in enumerate(leads):
        m = mask[:, :, li].astype(bool).any(axis=0)
        with np.errstate(invalid="ignore"):
            masked_prob = np.where(mask[:, :, li].astype(bool), pred_prob[:, :, li], np.nan)
            prob_mean = np.full(masked_prob.shape[1], 0.0)
            has_any = m
            if has_any.any():
                prob_mean[has_any] = np.nanmean(masked_prob[:, has_any], axis=0)
        flagged = np.where((prob_mean > threshold) & m)[0]
        if len(flagged) == 0:
            areas[lead] = []
            continue
        sub = coords[flagged]
        d = haversine_km(sub[:, None, :], sub[None, :, :])
        adj = (d <= link_km).astype(int)
        n_comp, labels = connected_components(csr_matrix(adj))
        clusters = []
        for cidx in range(n_comp):
            members = flagged[labels == cidx]
            if len(members) < min_cells:
                continue
            clusters.append(dict(
                n_cells=int(len(members)),
                mean_bust_prob=float(prob_mean[members].mean()),
                max_bust_prob=float(prob_mean[members].max()),
                centroid_lat=float(coords[members, 0].mean()),
                centroid_lon=float(coords[members, 1].mean()),
                bbox=[float(coords[members, 0].min()), float(coords[members, 0].max()),
                      float(coords[members, 1].min()), float(coords[members, 1].max())],
            ))
        areas[lead] = clusters
    return areas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()
    cfg = load_cfg()
    device = get_device(verbose=False)
    feat_dir = P(cfg, "features")
    meta = json.load(open(feat_dir / "feature_meta.json"))
    feature_cols = meta["feature_columns"]
    err_cols = [f"z_err_{v}" for v in meta["error_vars"]]
    leads = meta["leads"]

    ds, _ = build_datasets(feat_dir / "samples.parquet", feat_dir / "graph.npz",
                            feature_cols, err_cols, leads)
    hist_cols, lead_cols = split_feature_cols(feature_cols)
    g = np.load(feat_dir / "graph.npz")
    nbr = torch.from_numpy(g["nbr"]).long().to(device)
    edge_feat = torch.from_numpy(g["edge_feat"]).float().to(device)
    model = BustCastModel(n_hist_feat=len(hist_cols), n_lead_feat=len(lead_cols), n_points=len(g["point_id"]),
                           n_leads=len(leads), n_error_vars=len(err_cols), cfg=cfg,
                           coords_norm=norm_coords(g["region_centroid"]), region_pool=g["region_pool"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    loader = DataLoader(ds["test"], batch_size=4, collate_fn=collate)
    pred_err, pred_prob, true_err, mask, dates = run_inference(model, loader, nbr, edge_feat, device)

    # rebuild true bust labels aligned to the same (sample, point, lead) grid
    true_bust = np.zeros_like(true_err)
    df = pl.read_parquet(feat_dir / "samples.parquet").filter(pl.col("split") == "test")
    pid_to_slot = {int(p): i for i, p in enumerate(g["point_id"])}
    date_to_idx = {d: i for i, d in enumerate(dates)}
    for row in df.iter_rows(named=True):
        s, li = date_to_idx.get(str(row["init_date"])), leads.index(row["lead_day"]) if row["lead_day"] in leads else None
        slot = pid_to_slot.get(int(row["point_id"]))
        if s is None or li is None or slot is None:
            continue
        if row["bust"] is not None:
            true_bust[s, slot, li] = row["bust"]

    out_dir = ensure(P(cfg, "outputs") / "eval")
    rows = leadwise_metrics(pred_err, pred_prob, true_err, true_bust, mask, leads)
    pl.DataFrame(rows).write_csv(out_dir / "leadwise_metrics.csv")
    print(pl.DataFrame(rows))

    make_maps(g["coords"], g["point_id"], g["point_id"], true_err, pred_err, true_bust, pred_prob,
              mask, leads, dates, out_dir / "maps")

    region_aggregation(g["region_label"], None, true_err, pred_err, pred_prob, mask, leads,
                        out_dir / "region_aggregation.csv")

    areas = error_prone_areas(g["coords"], g["region_label"], pred_prob, mask, leads,
                               threshold=cfg["risk"]["edges"][2], min_cells=cfg["risk"]["min_cells"],
                               link_km=cfg["graph"]["link_km"])
    save_json(areas, out_dir / "error_prone_areas.json")
    print("Saved evaluation outputs ->", out_dir)


if __name__ == "__main__":
    main()
