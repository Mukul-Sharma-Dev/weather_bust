"""
Stage 1 baseline: XGBoost on the same engineered tabular features, one model per lead day
(Day 1..7) for both the error regression and the bust classification task. Also computes
Baseline 0 (historical/climatological error) for comparison. Everything is fit on
train(+val for early stopping) and evaluated on test; thresholds/medians reused from
data/features/feature_meta.json (fit on train/train+val only, see script 03).

Usage: python scripts/04_train_baseline_xgb.py
"""
import json, sys
from pathlib import Path
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bustcast.utils import load_cfg, P, ensure, save_json


def main():
    cfg = load_cfg()
    feat_dir = P(cfg, "features")
    out_dir = ensure(P(cfg, "outputs")) / "baselines"
    ensure(out_dir)
    meta = json.load(open(feat_dir / "feature_meta.json"))
    feature_cols = meta["feature_columns"]
    leads = meta["leads"]

    df = pl.read_parquet(feat_dir / "samples.parquet")
    # xgboost needs numeric-only, dense; fill nulls with the train median saved per column
    medians = {c: (df.filter(pl.col("split") == "train")[c].median() or 0.0)
               for c in feature_cols if c in df.columns}
    df = df.with_columns([pl.col(c).fill_null(medians[c]) for c in feature_cols if c in df.columns])

    device = "cuda" if cfg["xgb"]["device"] == "auto" and _has_cuda() else \
        ("cuda" if cfg["xgb"]["device"] == "cuda" else "cpu")
    print("XGBoost device:", device)

    results = {"error_regression": {}, "bust_classification": {}, "climatology_baseline": {}}
    for lead in leads:
        sub = df.filter(pl.col("lead_day") == lead)
        splits = {s: sub.filter(pl.col("split") == s) for s in ["train", "val", "test"]}
        if splits["train"].height == 0 or splits["test"].height == 0:
            print(f"lead {lead}: skipping (no train or test rows)"); continue
        Xtr, ytr = splits["train"].select(feature_cols).to_numpy(), splits["train"]["composite_error_z"].to_numpy()
        Xva, yva = splits["val"].select(feature_cols).to_numpy(), splits["val"]["composite_error_z"].to_numpy()
        Xte, yte = splits["test"].select(feature_cols).to_numpy(), splits["test"]["composite_error_z"].to_numpy()
        btr = splits["train"]["bust"].to_numpy()
        bva = splits["val"]["bust"].to_numpy()
        bte = splits["test"]["bust"].to_numpy()

        xc = cfg["xgb"]
        # --- regression head ---
        reg = xgb.XGBRegressor(max_depth=xc["max_depth"], learning_rate=xc["eta"],
                                subsample=xc["subsample"], colsample_bytree=xc["colsample_bytree"],
                                min_child_weight=xc["min_child_weight"], n_estimators=xc["n_rounds"],
                                objective="reg:pseudohubererror", tree_method="hist", device=device,
                                early_stopping_rounds=xc["early_stopping"])
        if Xva.shape[0] > 0:
            reg.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        else:
            reg.fit(Xtr, ytr)
        pred = reg.predict(Xte)
        mae = float(mean_absolute_error(yte, pred)) if len(yte) else None

        # --- classification head ---
        pos_w = max(1.0, (len(btr) - btr.sum()) / max(btr.sum(), 1))
        clf = xgb.XGBClassifier(max_depth=xc["max_depth"], learning_rate=xc["eta"],
                                 subsample=xc["subsample"], colsample_bytree=xc["colsample_bytree"],
                                 min_child_weight=xc["min_child_weight"], n_estimators=xc["n_rounds"],
                                 tree_method="hist", device=device, scale_pos_weight=pos_w,
                                 early_stopping_rounds=xc["early_stopping"], eval_metric="aucpr")
        if Xva.shape[0] > 0 and bva.sum() > 0:
            clf.fit(Xtr, btr, eval_set=[(Xva, bva)], verbose=False)
        else:
            clf.fit(Xtr, btr)
        proba = clf.predict_proba(Xte)[:, 1] if len(Xte) else np.array([])
        roc = float(roc_auc_score(bte, proba)) if len(np.unique(bte)) > 1 else None
        pr = float(average_precision_score(bte, proba)) if len(np.unique(bte)) > 1 else None

        # --- Baseline 0: historical climatological error (train-mean error for this lead+zone) ---
        clim = splits["train"].group_by("zone").agg(pl.col("composite_error_z").mean().alias("clim_pred"))
        clim_map = {r["zone"]: r["clim_pred"] for r in clim.to_dicts()}
        clim_pred = np.array([clim_map.get(z, 0.0) for z in splits["test"]["zone"].to_list()])
        clim_mae = float(mean_absolute_error(yte, clim_pred)) if len(yte) else None

        results["error_regression"][lead] = {"mae": mae, "n_test": int(len(yte))}
        results["bust_classification"][lead] = {"roc_auc": roc, "pr_auc": pr, "n_test": int(len(bte)),
                                                  "bust_rate_test": float(bte.mean()) if len(bte) else None}
        results["climatology_baseline"][lead] = {"mae": clim_mae}

        reg.save_model(str(out_dir / f"xgb_reg_lead{lead}.json"))
        clf.save_model(str(out_dir / f"xgb_clf_lead{lead}.json"))
        print(f"lead {lead}: reg MAE={mae}, clim MAE={clim_mae}, ROC-AUC={roc}, PR-AUC={pr}")

    save_json(results, out_dir / "xgb_baseline_metrics.json")
    save_json(medians, out_dir / "xgb_feature_medians.json")
    print("Saved ->", out_dir / "xgb_baseline_metrics.json")


def _has_cuda():
    try:
        import subprocess
        return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    except Exception:
        return False


if __name__ == "__main__":
    main()
