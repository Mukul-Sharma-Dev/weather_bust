"""
Stage 2/3 training of the BustCastModel.

Stage 2 (--stage 2): continuous error regression only (Huber loss on the composite error).
Stage 3 (--stage 3): loads the Stage-2 checkpoint and adds the bust-probability head
                      (class-weighted BCE / focal loss), fine-tuning the whole network.

Ablations (--ablation A|B|C|D|E) zero out feature groups per the spec's Sec.26 table:
  A: GFS only                       -> drop hist(ERA5 temporal) and spatial features
  B: GFS + ERA5 temporal            -> drop spatial-neighbourhood features
  C: GFS + spatial                  -> drop ERA5 temporal features
  D: GFS + temporal + spatial       -> keep both, drop forecast-cycle-revision features
  E: full model                     -> everything, including forecast-cycle disagreement

Usage:
    python scripts/05_train_transformer.py --stage 2
    python scripts/05_train_transformer.py --stage 3 --init_from outputs/checkpoints/stage2_best.pt
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bustcast.utils import load_cfg, P, ensure, save_json, set_seed, get_device
from bustcast.dataset import build_datasets, collate, split_feature_cols, HIST_SUFFIXES, LEAD_SUFFIXES
from bustcast.model import BustCastModel
from bustcast.geo import norm_coords

ABLATIONS = {
    "A": {"drop_hist": True, "drop_spatial": True, "drop_cyclechange": True},
    "B": {"drop_hist": False, "drop_spatial": True, "drop_cyclechange": True},
    "C": {"drop_hist": True, "drop_spatial": False, "drop_cyclechange": True},
    "D": {"drop_hist": False, "drop_spatial": False, "drop_cyclechange": True},
    "E": {"drop_hist": False, "drop_spatial": False, "drop_cyclechange": False},
}


def filter_features(feature_cols, ablation):
    if ablation is None:
        return feature_cols
    rule = ABLATIONS[ablation]
    spatial_tags = ("_localmean", "_localstd", "_anomaly", "_neighbdisagree")
    hist_tags = ("_prev1", "_delta1", "_delta2", "_rollmean3", "_rollstd3", "_rollmean7", "_rollstd7")
    keep = []
    for c in feature_cols:
        if rule["drop_spatial"] and c.endswith(spatial_tags):
            continue
        if rule["drop_hist"] and c.endswith(hist_tags):
            continue
        if rule["drop_cyclechange"] and c.endswith("_cyclechange"):
            continue
        keep.append(c)
    return keep


def move(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def focal_loss(logits, target, gamma):
    p = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, p, 1 - p)
    return (-(1 - pt).pow(gamma) * torch.log(pt.clamp_min(1e-6))).mean()


def run_epoch(model, loader, nbr, edge_feat, device, optimizer, tc, stage, scaler, train=True):
    model.train(train)
    tot_loss = tot_reg = tot_cls = 0.0
    n_batches = 0
    for batch in loader:
        batch = move(batch, device)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=(tc["amp"] and device.type == "cuda")):
                err, errvar, bust_logit = model(batch["Xh"], batch["Xl"], nbr, edge_feat)
                mask = batch["mask"]
                huber = F.huber_loss(err, batch["Ycomp"], delta=tc["huber_delta"], reduction="none")
                reg_loss = (huber * mask).sum() / mask.sum().clamp_min(1)
                errvar_huber = F.huber_loss(errvar, batch["Yerr"], delta=tc["huber_delta"], reduction="none")
                reg_loss = reg_loss + tc["aux_weight"] * (errvar_huber * mask[..., None]).sum() / \
                    mask[..., None].sum().clamp_min(1)

                cls_loss = torch.tensor(0.0, device=device)
                if stage >= 3:
                    if tc["bust_loss"] == "focal":
                        cls_loss = focal_loss(bust_logit[mask.bool()], batch["Ybust"][mask.bool()],
                                               tc["focal_gamma"])
                    else:
                        pos = batch["Ybust"][mask.bool()]
                        pos_w = ((1 - pos.mean()) / pos.mean().clamp_min(1e-3)).clamp(1, 50)
                        cls_loss = F.binary_cross_entropy_with_logits(
                            bust_logit[mask.bool()], pos, pos_weight=pos_w)

                loss = tc["lambda_err"] * reg_loss + (tc["lambda_bust"] * cls_loss if stage >= 3 else 0.0)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
                scaler.step(optimizer)
                scaler.update()

        tot_loss += loss.item(); tot_reg += reg_loss.item(); tot_cls += cls_loss.detach().item()
        n_batches += 1
    n_batches = max(n_batches, 1)
    return dict(loss=tot_loss / n_batches, reg=tot_reg / n_batches, cls=tot_cls / n_batches)


def make_loader(ds, batch_size, shuffle):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, drop_last=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[2, 3], default=2)
    ap.add_argument("--init_from", default=None)
    ap.add_argument("--ablation", default=None, choices=list(ABLATIONS.keys()))
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_cfg()
    set_seed(cfg["train"]["seed"])
    device = get_device()

    feat_dir = P(cfg, "features")
    meta = json.load(open(feat_dir / "feature_meta.json"))
    feature_cols = filter_features(meta["feature_columns"], args.ablation)
    err_vars = meta["error_vars"]
    err_cols = [f"z_err_{v}" for v in err_vars]
    leads = meta["leads"]

    ds, medians = build_datasets(feat_dir / "samples.parquet", feat_dir / "graph.npz",
                                  feature_cols, err_cols, leads)
    hist_cols, lead_cols = split_feature_cols(feature_cols)
    print(f"features: {len(feature_cols)} (hist={len(hist_cols)}, lead={len(lead_cols)})"
          + (f"  [ablation {args.ablation}]" if args.ablation else ""))

    g = np.load(feat_dir / "graph.npz")
    nbr = torch.from_numpy(g["nbr"]).long().to(device)
    edge_feat = torch.from_numpy(g["edge_feat"]).float().to(device)
    region_pool = g["region_pool"]
    region_xy = norm_coords(g["region_centroid"])

    model = BustCastModel(n_hist_feat=len(hist_cols), n_lead_feat=len(lead_cols), n_points=len(g["point_id"]),
                           n_leads=len(leads), n_error_vars=len(err_cols), cfg=cfg,
                           coords_norm=region_xy, region_pool=region_pool).to(device)
    if args.init_from:
        model.load_state_dict(torch.load(args.init_from, map_location=device), strict=False)
        print("Initialised from", args.init_from)

    tc = dict(cfg["train"])
    if args.stage == 3:
        tc["lr"] = tc["lr_stage3"]
    epochs = args.epochs or (tc["epochs_stage2"] if args.stage == 2 else tc["epochs_stage3"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    warmup = tc["warmup_epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: min(1.0, (e + 1) / max(warmup, 1)) *
        (0.5 * (1 + np.cos(np.pi * max(0, e - warmup) / max(epochs - warmup, 1)))))
    scaler = torch.amp.GradScaler(enabled=(tc["amp"] and device.type == "cuda"))

    batch_size = tc["batch_size"]
    ckpt_dir = ensure(Path(cfg["paths"]["outputs"]) / "checkpoints")
    tag = f"stage{args.stage}" + (f"_ablation{args.ablation}" if args.ablation else "")

    best_val, patience_ct = float("inf"), 0
    history = []
    for epoch in range(epochs):
        t0 = time.time()
        while True:
            try:
                train_loader = make_loader(ds["train"], batch_size, True)
                tr = run_epoch(model, train_loader, nbr, edge_feat, device, optimizer, tc, args.stage,
                                scaler, train=True)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                print(f"CUDA OOM -> halving batch size to {batch_size}")
        val_loader = make_loader(ds["val"], batch_size, False)
        va = run_epoch(model, val_loader, nbr, edge_feat, device, optimizer, tc, args.stage, scaler,
                        train=False)
        sched.step()
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[{tag}] epoch {epoch+1}/{epochs}  train={tr['loss']:.4f} (reg={tr['reg']:.4f} cls={tr['cls']:.4f})  "
              f"val={va['loss']:.4f}  lr={lr_now:.2e}  {dt:.1f}s"
              + (f"  GPU_mem={torch.cuda.max_memory_allocated()/2**20:.0f}MB" if device.type == "cuda" else ""))
        history.append({"epoch": epoch + 1, **{f"train_{k}": v for k, v in tr.items()},
                         **{f"val_{k}": v for k, v in va.items()}, "lr": lr_now})

        if va["loss"] < best_val - 1e-5:
            best_val, patience_ct = va["loss"], 0
            torch.save(model.state_dict(), ckpt_dir / f"{tag}_best.pt")
        else:
            patience_ct += 1
            if patience_ct >= tc["patience"]:
                print(f"Early stopping at epoch {epoch+1} (patience {tc['patience']})")
                break

    torch.save(model.state_dict(), ckpt_dir / f"{tag}_final.pt")
    save_json({"history": history, "feature_columns": feature_cols, "ablation": args.ablation,
               "best_val_loss": best_val}, ckpt_dir / f"{tag}_train_log.json")
    save_json(medians, ckpt_dir / f"{tag}_feature_medians.json")
    print(f"Saved checkpoints -> {ckpt_dir}/{tag}_{{best,final}}.pt")


if __name__ == "__main__":
    main()
