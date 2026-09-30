"""
Two complementary explanation layers (spec Sec.23 explicitly warns not to treat attention
weights alone as definitive -- neither of these relies on attention):

1. Permutation importance on the XGBoost baseline, per lead day -> which engineered features
   generally matter for error/bust prediction. Global, model-level.

2. Input-gradient saliency on the Transformer's bust-probability output, computed only for
   points/leads the model already flagged as low-confidence (P(bust) above the risk threshold)
   -> a per-flagged-point ranked list of contributing input features, mapped to the
   plain-language factor names from Sec.23 (pressure tendency, wind disagreement, humidity
   anomaly, forecast-cycle revision, historical error level, spatial gradient). This is a
   real gradient computed from the actual model and the actual input, not hard-coded text --
   only the phrase mapping (column-name -> human label) is fixed.

Usage: python scripts/09_explainability.py --checkpoint outputs/checkpoints/stage3_best.pt
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import polars as pl
import torch
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bustcast.utils import load_cfg, P, ensure, save_json, get_device
from bustcast.dataset import build_datasets, collate, split_feature_cols
from bustcast.model import BustCastModel
from bustcast.geo import norm_coords

from bustcast.explain import to_factor


from sklearn.base import BaseEstimator, RegressorMixin

class XGBWrapper(BaseEstimator, RegressorMixin):
    """sklearn permutation_importance needs .fit()/.predict(); wrap a pre-trained booster."""
    def __init__(self, booster=None):
        self.booster = booster

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.booster.predict(X)


def xgb_permutation_importance(cfg, feat_dir, out_dir):
    import xgboost as xgb
    meta = json.load(open(feat_dir / "feature_meta.json"))
    feature_cols = meta["feature_columns"]
    df = pl.read_parquet(feat_dir / "samples.parquet")
    medians = {c: (df.filter(pl.col("split") == "train")[c].median() or 0.0) for c in feature_cols}
    df = df.with_columns([pl.col(c).fill_null(medians[c]) for c in feature_cols])

    baselines_dir = Path(cfg["paths"]["outputs"]) / "baselines"
    results = {}
    for lead in meta["leads"]:
        model_path = baselines_dir / f"xgb_reg_lead{lead}.json"
        if not model_path.exists():
            continue
        booster = xgb.XGBRegressor()
        booster.load_model(str(model_path))
        test = df.filter((pl.col("split") == "test") & (pl.col("lead_day") == lead))
        if test.height < 10:
            continue
        X, y = test.select(feature_cols).to_numpy(), test["composite_error_z"].to_numpy()
        pi = permutation_importance(XGBWrapper(booster), X, y, n_repeats=5, random_state=0,
                                     scoring="neg_mean_absolute_error")
        order = np.argsort(-pi.importances_mean)[:10]
        results[lead] = [{"feature": feature_cols[i], "factor": to_factor(feature_cols[i]),
                           "importance": float(pi.importances_mean[i])} for i in order]
    save_json(results, out_dir / "xgb_permutation_importance.json")
    return results


def transformer_saliency(cfg, feat_dir, checkpoint, out_dir, prob_threshold):
    meta = json.load(open(feat_dir / "feature_meta.json"))
    feature_cols = meta["feature_columns"]
    err_cols = [f"z_err_{v}" for v in meta["error_vars"]]
    leads = meta["leads"]
    device = get_device(verbose=False)

    ds, _ = build_datasets(feat_dir / "samples.parquet", feat_dir / "graph.npz",
                            feature_cols, err_cols, leads)
    hist_cols, lead_cols = split_feature_cols(feature_cols)
    g = np.load(feat_dir / "graph.npz")
    nbr = torch.from_numpy(g["nbr"]).long().to(device)
    edge_feat = torch.from_numpy(g["edge_feat"]).float().to(device)
    model = BustCastModel(n_hist_feat=len(hist_cols), n_lead_feat=len(lead_cols), n_points=len(g["point_id"]),
                           n_leads=len(leads), n_error_vars=len(err_cols), cfg=cfg,
                           coords_norm=norm_coords(g["region_centroid"]), region_pool=g["region_pool"]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    explanations = []
    n_scanned = 0
    for i in range(len(ds["test"])):
        item = ds["test"][i]
        Xh = item["Xh"].unsqueeze(0).to(device).requires_grad_(True)
        Xl = item["Xl"].unsqueeze(0).to(device).requires_grad_(True)
        mask = item["mask"].numpy()
        _, _, bust_logit = model(Xh, Xl, nbr, edge_feat)
        prob = torch.sigmoid(bust_logit)[0]              # [N,L]
        flagged = (prob.detach().cpu().numpy() > prob_threshold) & mask.astype(bool)
        pts, ls = np.where(flagged)
        for p_idx, l_idx in zip(pts[:3], ls[:3]):         # cap per-sample to keep this fast
            model.zero_grad(set_to_none=True)
            if Xh.grad is not None: Xh.grad = None
            if Xl.grad is not None: Xl.grad = None
            prob[p_idx, l_idx].backward(retain_graph=True)
            sal_h = (Xh.grad[0, p_idx].abs() * Xh[0, p_idx].abs()).detach().cpu().numpy()
            sal_l = (Xl.grad[0, p_idx, l_idx].abs() * Xl[0, p_idx, l_idx].abs()).detach().cpu().numpy()
            top_h = np.argsort(-sal_h)[:4]
            top_l = np.argsort(-sal_l)[:3]
            factors = [to_factor(hist_cols[j]) for j in top_h] + [to_factor(lead_cols[j]) for j in top_l]
            seen, ranked = set(), []
            for f in factors:
                if f not in seen:
                    ranked.append(f); seen.add(f)
            explanations.append(dict(
                init_date=item["init_date"], point_id=int(g["point_id"][p_idx]),
                lead_day=leads[l_idx], bust_probability=float(prob[p_idx, l_idx].item()),
                dominant_factors=ranked[:5],
            ))
        n_scanned += 1
        if n_scanned >= 40:      # bound the scan; the API computes this on-demand for a single init anyway
            break
    save_json(explanations, out_dir / "sample_explanations.json")
    return explanations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prob_threshold", type=float, default=0.5)
    args = ap.parse_args()
    cfg = load_cfg()
    feat_dir = P(cfg, "features")
    out_dir = ensure(P(cfg, "outputs") / "explainability")

    print("Computing XGBoost permutation importance ...")
    pi = xgb_permutation_importance(cfg, feat_dir, out_dir)
    for lead, feats in list(pi.items())[:2]:
        print(f"  lead {lead} top factors:", [f["factor"] for f in feats[:5]])

    print("Computing Transformer input-gradient saliency for flagged points ...")
    ex = transformer_saliency(cfg, feat_dir, args.checkpoint, out_dir, args.prob_threshold)
    print(f"  wrote {len(ex)} example explanations ->", out_dir / "sample_explanations.json")


if __name__ == "__main__":
    main()
