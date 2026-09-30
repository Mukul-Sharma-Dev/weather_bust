"""
Fits probability calibration on the dedicated `calib` split (never train/val/test), so raw
network sigmoid outputs are not assumed to already be calibrated probabilities (spec Sec.15).
Tries none / Platt (logistic) / isotonic per lead day, picks whichever minimises Brier score
on the calib split, and reports Brier / log-loss / reliability-curve points / ECE on `test`.

Usage: python scripts/06_calibrate.py --checkpoint outputs/checkpoints/stage3_best.pt
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bustcast.utils import load_cfg, P, ensure, save_json, get_device
from bustcast.dataset import build_datasets, collate, split_feature_cols
from bustcast.model import BustCastModel
from bustcast.geo import norm_coords
from torch.utils.data import DataLoader


def ece(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(probs, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return float(e)


def reliability_curve(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(probs, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    pts = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        pts.append({"bin_mid": float((bins[b] + bins[b + 1]) / 2), "n": int(m.sum()),
                    "mean_pred": float(probs[m].mean()), "mean_obs": float(labels[m].mean())})
    return pts


def collect_predictions(model, loader, nbr, edge_feat, device, mask_split_leads):
    all_logit, all_bust, all_mask, all_lead = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            _, _, bust_logit = model(batch["Xh"], batch["Xl"], nbr, edge_feat)
            all_logit.append(bust_logit.cpu().numpy())
            all_bust.append(batch["Ybust"].cpu().numpy())
            all_mask.append(batch["mask"].cpu().numpy())
    logit = np.concatenate(all_logit, 0)     # [B,N,L]
    bust = np.concatenate(all_bust, 0)
    mask = np.concatenate(all_mask, 0)
    return logit, bust, mask


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

    calib_logit, calib_bust, calib_mask = collect_predictions(
        model, DataLoader(ds["calib"], batch_size=4, collate_fn=collate), nbr, edge_feat, device, None)
    test_logit, test_bust, test_mask = collect_predictions(
        model, DataLoader(ds["test"], batch_size=4, collate_fn=collate), nbr, edge_feat, device, None)

    out_dir = ensure(P(cfg, "outputs") / "calibration")
    results = {}
    calibrators = {}
    for li, lead in enumerate(leads):
        cm = calib_mask[:, :, li].astype(bool)
        if cm.sum() < 20:
            results[lead] = {"note": "too few calib samples, using raw sigmoid"}
            continue
        c_logit, c_y = calib_logit[:, :, li][cm], calib_bust[:, :, li][cm]
        c_prob_raw = 1 / (1 + np.exp(-c_logit))

        tm = test_mask[:, :, li].astype(bool)
        t_logit, t_y = test_logit[:, :, li][tm], test_bust[:, :, li][tm]
        t_prob_raw = 1 / (1 + np.exp(-t_logit))

        methods = {}
        methods["none"] = (t_prob_raw, None)
        if len(np.unique(c_y)) > 1:
            platt = LogisticRegression().fit(c_logit.reshape(-1, 1), c_y)
            methods["platt"] = (platt.predict_proba(t_logit.reshape(-1, 1))[:, 1], platt)
            iso = IsotonicRegression(out_of_bounds="clip").fit(c_prob_raw, c_y)
            methods["isotonic"] = (iso.predict(t_prob_raw), iso)

        # pick method by Brier score ON CALIB (not test) to choose, then report test metrics
        calib_briers = {}
        for name, (_, fit) in methods.items():
            if name == "none":
                calib_briers[name] = brier_score_loss(c_y, c_prob_raw)
            elif name == "platt":
                calib_briers[name] = brier_score_loss(c_y, fit.predict_proba(c_logit.reshape(-1, 1))[:, 1])
            else:
                calib_briers[name] = brier_score_loss(c_y, fit.predict(c_prob_raw))
        best = min(calib_briers, key=calib_briers.get)
        t_prob_best = methods[best][0]
        t_prob_best = np.clip(t_prob_best, 1e-4, 1 - 1e-4)

        results[lead] = dict(
            chosen_method=best, calib_brier_by_method=calib_briers,
            test_brier=float(brier_score_loss(t_y, t_prob_best)) if len(t_y) else None,
            test_logloss=float(log_loss(t_y, t_prob_best, labels=[0, 1])) if len(t_y) else None,
            test_ece=ece(t_prob_best, t_y) if len(t_y) else None,
            reliability=reliability_curve(t_prob_best, t_y) if len(t_y) else [],
            n_test=int(len(t_y)),
        )
        calibrators[lead] = best
        print(f"lead {lead}: chosen={best}  test_brier={results[lead]['test_brier']}  "
              f"test_ece={results[lead]['test_ece']}")

    save_json(results, out_dir / "calibration_report.json")
    save_json(cfg["risk"], out_dir / "risk_bins.json")
    print("Saved ->", out_dir / "calibration_report.json")


if __name__ == "__main__":
    main()
