"""
Runs the Sec.26 ablation ladder end to end:
    A: GFS only
    B: GFS + ERA5 temporal history
    C: GFS + spatial neighbourhood
    D: GFS + temporal + spatial
    E: full model (+ forecast-cycle disagreement)

Each ablation trains a short Stage-2-only model (fewer epochs than a full run -- this is a
diagnostic, not the final model) and reports test MAE/ROC-AUC/PR-AUC averaged over leads, so
you can see which information sources actually move the needle before committing to the full
Stage 2 -> Stage 3 -> calibration run on the real dataset.

Usage: python scripts/08_ablations.py --epochs 15
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bustcast.utils import load_cfg, P, ensure, save_json, get_device
from bustcast.dataset import build_datasets, collate, split_feature_cols
from bustcast.model import BustCastModel
from bustcast.geo import norm_coords
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
train_mod = importlib.import_module("05_train_transformer")


def evaluate_ablation(model, ds, feat_dir, nbr, edge_feat, device, leads, err_cols):
    loader = DataLoader(ds["test"], batch_size=4, collate_fn=collate)
    maes, aucs, praucs = [], [], []
    model.eval()
    with torch.no_grad():
        E, Y, M, B, P_ = [], [], [], [], []
        for batch in loader:
            b2 = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            err, _, bust_logit = model(b2["Xh"], b2["Xl"], nbr, edge_feat)
            E.append(err.cpu().numpy()); Y.append(batch["Ycomp"].numpy())
            M.append(batch["mask"].numpy()); B.append(batch["Ybust"].numpy())
            P_.append(torch.sigmoid(bust_logit).cpu().numpy())
    E, Y, M, B, P_ = map(lambda a: np.concatenate(a, 0), (E, Y, M, B, P_))
    for li in range(len(leads)):
        m = M[:, :, li].astype(bool)
        if m.sum() == 0:
            continue
        maes.append(mean_absolute_error(Y[:, :, li][m], E[:, :, li][m]))
        tb = B[:, :, li][m]
        if len(np.unique(tb)) > 1:
            aucs.append(roc_auc_score(tb, P_[:, :, li][m]))
            praucs.append(average_precision_score(tb, P_[:, :, li][m]))
    return dict(mean_MAE=float(np.mean(maes)) if maes else None,
                mean_ROC_AUC=float(np.mean(aucs)) if aucs else None,
                mean_PR_AUC=float(np.mean(praucs)) if praucs else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()
    cfg = load_cfg()
    device = get_device()
    feat_dir = P(cfg, "features")
    meta = json.load(open(feat_dir / "feature_meta.json"))
    err_cols = [f"z_err_{v}" for v in meta["error_vars"]]
    leads = meta["leads"]

    results = {}
    for ablation in ["A", "B", "C", "D", "E"]:
        print(f"\n=== Ablation {ablation} ===")
        subprocess.run([sys.executable, str(Path(__file__).parent / "05_train_transformer.py"),
                         "--stage", "2", "--ablation", ablation, "--epochs", str(args.epochs)],
                        check=True)
        tag = f"stage2_ablation{ablation}"
        ckpt = Path(cfg["paths"]["outputs"]) / "checkpoints" / f"{tag}_best.pt"
        feature_cols = train_mod.filter_features(meta["feature_columns"], ablation)
        hist_cols, lead_cols = split_feature_cols(feature_cols)
        ds, _ = build_datasets(feat_dir / "samples.parquet", feat_dir / "graph.npz",
                                feature_cols, err_cols, leads)
        g = np.load(feat_dir / "graph.npz")
        nbr = torch.from_numpy(g["nbr"]).long().to(device)
        edge_feat = torch.from_numpy(g["edge_feat"]).float().to(device)
        model = BustCastModel(n_hist_feat=len(hist_cols), n_lead_feat=len(lead_cols),
                               n_points=len(g["point_id"]), n_leads=len(leads),
                               n_error_vars=len(err_cols), cfg=cfg,
                               coords_norm=norm_coords(g["region_centroid"]),
                               region_pool=g["region_pool"]).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        results[ablation] = evaluate_ablation(model, ds, feat_dir, nbr, edge_feat, device, leads, err_cols)
        print(ablation, results[ablation])

    out_dir = ensure(P(cfg, "outputs") / "ablations")
    save_json(results, out_dir / "ablation_results.json")
    print("\nSaved ->", out_dir / "ablation_results.json")
    print(pl.DataFrame([{"ablation": k, **v} for k, v in results.items()]))


if __name__ == "__main__":
    main()
