"""Shared by scripts/09_explainability.py (offline batch report) and app/api.py (on-demand,
single init_date). See scripts/09 for the full docstring on what these two explanation layers
are and are not (never attention-weights-as-explanation; see spec Sec.23)."""
import numpy as np
import torch

FACTOR_MAP = [
    (("pressure_msl", "surface_pressure"), "pressure tendency"),
    (("wind_speed", "wind_gusts", "wind10", "wind100"), "wind change / disagreement"),
    (("relative_humidity",), "humidity anomaly"),
    (("precipitation", "rain"), "precipitation anomaly"),
    (("temperature_2m",), "temperature tendency"),
    (("cyclechange",), "forecast-cycle disagreement (GFS revision)"),
    (("localstd", "neighbdisagree", "anomaly"), "spatial neighbour disagreement"),
    (("rollstd", "rollmean"), "recent variability / historical error level"),
]


def to_factor(colname):
    for keys, label in FACTOR_MAP:
        if any(k in colname for k in keys):
            return label
    return colname


def explain_for_request(state, Xh, Xl, hist_done, prob_threshold, max_points=15):
    """Gradient x input saliency for every (point, lead) the model currently flags, computed
    live against this request's own features -- not a cached/precomputed explanation."""
    device = state["device"]
    model, nbr, edge_feat = state["model"], state["nbr"], state["edge_feat"]
    hist_cols, lead_cols, leads = state["hist_cols"], state["lead_cols"], state["leads"]

    Xh_t = torch.from_numpy(Xh).unsqueeze(0).to(device).requires_grad_(True)
    Xl_t = torch.from_numpy(Xl).unsqueeze(0).to(device).requires_grad_(True)
    _, _, bust_logit = model(Xh_t, Xl_t, nbr, edge_feat)
    prob = torch.sigmoid(bust_logit)[0]
    prob_np = prob.detach().cpu().numpy()
    flagged = (prob_np > prob_threshold) & hist_done[:, None]
    pts, ls = np.where(flagged)

    out = []
    for p_idx, l_idx in list(zip(pts, ls))[:max_points]:
        model.zero_grad(set_to_none=True)
        if Xh_t.grad is not None: Xh_t.grad = None
        if Xl_t.grad is not None: Xl_t.grad = None
        prob[p_idx, l_idx].backward(retain_graph=True)
        sal_h = (Xh_t.grad[0, p_idx].abs() * Xh_t[0, p_idx].abs()).detach().cpu().numpy()
        sal_l = (Xl_t.grad[0, p_idx, l_idx].abs() * Xl_t[0, p_idx, l_idx].abs()).detach().cpu().numpy()
        top_h = np.argsort(-sal_h)[:4]
        top_l = np.argsort(-sal_l)[:3]
        factors, seen = [], set()
        for j in top_h:
            f = to_factor(hist_cols[j])
            if f not in seen: factors.append(f); seen.add(f)
        for j in top_l:
            f = to_factor(lead_cols[j])
            if f not in seen: factors.append(f); seen.add(f)
        out.append(dict(point_id=int(state["g"]["point_id"][p_idx]), lead_day=leads[l_idx],
                         bust_probability=float(prob_np[p_idx, l_idx]), dominant_factors=factors[:5]))
    return out
