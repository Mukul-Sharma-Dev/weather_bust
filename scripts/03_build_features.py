"""
Turns gfs_daily.parquet + era5_daily.parquet into one long feature table:
one row per (point_id, init_date, lead_day in 1..7), with:

  - GFS forecast state at that lead (legitimate input: known at init time)
  - forecast-cycle revision  = GFS(this init) - GFS(init-1day, lead+1) for the same valid date
  - ERA5 temporal history features (previous_value, delta_1/2, rolling mean/std) for the
    `context_days` days strictly BEFORE init_date -- never anything at or after init_date
  - spatial neighbourhood features (kNN graph, since script 01/geo.detect_grid found the
    900 points are not a regular lat/lon grid) built from the last observed ERA5 state
    (t0-1), shared by all 7 leads of that init
  - per-variable error = |GFS - ERA5| at the forecast valid date, plus a normalised
    composite error and a region/lead-dependent bust label

All normalisation / threshold statistics are fit on TRAIN (+VAL where noted) only, and the
fitted parameters are saved to data/features/feature_meta.json so evaluation never touches
train-derived numbers computed from the test split.

Output: data/features/samples.parquet  (long format)
        data/features/feature_meta.json
        data/features/graph.npz        (kNN indices/edges/gradient operator, region pooling)
"""
import argparse, sys
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bustcast.utils import load_cfg, P, ensure, save_json
from bustcast import geo

EPS = 1e-6


def build_graph(points_df, cfg):
    coords = points_df.select(["latitude", "longitude"]).to_numpy()
    k, k_stats = geo.choose_k(coords, cfg["graph"])
    nbr, dist = geo.knn(coords, k)
    east, north = geo.local_offsets(coords, nbr)
    G = geo.gradient_operator(east, north, ridge=cfg["graph"]["grad_ridge"])
    edge_f = geo.edge_features(east, north, dist)
    reg_lab, reg_cent, reg_pool = geo.make_regions(coords, cfg["graph"]["n_regions"])
    grid_info = geo.detect_grid(coords)
    np.savez(ensure(P(cfg, "features")) / "graph.npz",
             nbr=nbr, dist=dist, grad_op=G, edge_feat=edge_f, region_label=reg_lab,
             region_centroid=reg_cent, region_pool=reg_pool, k=k, coords=coords,
             point_id=points_df["id"].to_numpy())
    return dict(spatial_representation="graph_attention (irregular points)",
                k=int(k), k_search=k_stats, n_regions=int(reg_pool.shape[0]),
                grid_detection=grid_info)


def temporal_history(era5, cfg):
    c = cfg["columns"]
    ctx = cfg["features"]["context_days"]
    windows = cfg["features"]["hist_windows"]
    vars_ = [v for v in cfg["features"]["core_vars"] if v in era5.columns]
    era5 = era5.sort([c["id"], "date"])

    exprs = []
    for v in vars_:
        s1 = pl.col(v).shift(1).over(c["id"])
        s2 = pl.col(v).shift(2).over(c["id"])
        exprs += [
            s1.alias(f"{v}_prev1"),
            (s1 - s2).alias(f"{v}_delta1"),
            (s1 - pl.col(v).shift(3).over(c["id"])).alias(f"{v}_delta2"),
        ]
        for w in windows:
            shifted = [pl.col(v).shift(i).over(c["id"]) for i in range(1, w + 1)]
            mean_w = pl.mean_horizontal(shifted).alias(f"{v}_rollmean{w}")
            # population std across the w shifted lag columns (constant small w -> compute manually)
            m = pl.mean_horizontal(shifted)
            var_terms = [(s - m) ** 2 for s in shifted]
            std_w = (pl.mean_horizontal(var_terms)).sqrt().alias(f"{v}_rollstd{w}")
            exprs += [mean_w, std_w]
    era5 = era5.with_columns(exprs)
    # valid_date of this history snapshot == the init_date it will be attached to
    era5 = era5.rename({"date": "hist_asof_date"})
    keep = [c["id"], "hist_asof_date"] + [e.meta.output_name() for e in exprs]
    return era5.select(keep)


def gfs_revision(gfs, cfg):
    """GFS(this init) - GFS(init-1day, lead+1) for the same valid_date & point."""
    c = cfg["columns"]
    cur = gfs.select([c["id"], "valid_date", c["lead"], "init_date"] +
                      [col for col in gfs.columns if col.endswith(("_mean", "_sum", "_max", "_min"))])
    prev = cur.rename({col: f"{col}__prevcycle" for col in cur.columns
                        if col not in (c["id"], "valid_date")})
    prev = prev.with_columns([
        (pl.col("init_date__prevcycle") + pl.duration(days=1)).alias("init_date"),
        (pl.col(f"{c['lead']}__prevcycle") - 1).alias(c["lead"]),
    ])
    joined = cur.join(prev, on=[c["id"], "valid_date", "init_date", c["lead"]], how="left")
    val_cols = [col for col in cur.columns if col.endswith(("_mean", "_sum", "_max", "_min"))]
    for col in val_cols:
        joined = joined.with_columns(
            (pl.col(col) - pl.col(f"{col}__prevcycle")).alias(f"{col}_cyclechange")
        )
    return joined.select([c["id"], "init_date", c["lead"]] + [f"{v}_cyclechange" for v in val_cols])


def spatial_features(era5_asof, cfg, graph_npz):
    """Local mean/std/anomaly/gradient of the last-observed ERA5 state, per init_date."""
    c = cfg["columns"]
    vars_ = [v for v in cfg["features"]["spatial_vars"] if v in era5_asof.columns]
    pid_to_idx = {int(pid): i for i, pid in enumerate(graph_npz["point_id"])}
    nbr = graph_npz["nbr"]

    dates = era5_asof["hist_asof_date"].unique().sort().to_list()
    out_rows = []
    piv = era5_asof.select([c["id"], "hist_asof_date"] + vars_)
    for d in dates:
        day = piv.filter(pl.col("hist_asof_date") == d)
        if day.height < 2:
            continue
        idx_map = {int(pid): i for i, pid in enumerate(day[c["id"]].to_list())}
        vals = day.select(vars_).to_numpy()
        pid_list = day[c["id"]].to_list()
        row_dict = {c["id"]: pid_list, "hist_asof_date": [d] * len(pid_list)}
        for vi, v in enumerate(vars_):
            x = vals[:, vi]
            neigh_vals = np.full((len(pid_list), nbr.shape[1]), np.nan)
            for r, pid in enumerate(pid_list):
                gidx = pid_to_idx.get(int(pid))
                if gidx is None:
                    continue
                for kk, nb_gidx in enumerate(nbr[gidx]):
                    nb_pid = int(graph_npz["point_id"][nb_gidx])
                    j = idx_map.get(nb_pid)
                    if j is not None:
                        neigh_vals[r, kk] = vals[j, vi]
            local_mean = np.nanmean(neigh_vals, axis=1)
            local_std = np.nanstd(neigh_vals, axis=1)
            row_dict[f"{v}_localmean"] = local_mean
            row_dict[f"{v}_localstd"] = local_std
            row_dict[f"{v}_anomaly"] = x - local_mean
            row_dict[f"{v}_neighbdisagree"] = local_std
        out_rows.append(pl.DataFrame(row_dict))
    return pl.concat(out_rows, how="vertical_relaxed") if out_rows else None


def composite_error_and_bust(df, cfg, train_mask_col):
    """z-normalise each variable's abs-error using TRAIN rows only, average -> composite error,
    then threshold per (zone, lead) on TRAIN+VAL only (never test) to build the bust label."""
    tvars = [v for v in cfg["targets"]["variables"] if f"{v}_gfs" in df.columns and f"{v}_era5" in df.columns]
    err_cols = []
    stats = {}
    for v in tvars:
        ecol = f"err_{v}"
        df = df.with_columns((pl.col(f"{v}_gfs") - pl.col(f"{v}_era5")).abs().alias(ecol))
        mu = df.filter(pl.col(train_mask_col))[ecol].mean()
        sd = df.filter(pl.col(train_mask_col))[ecol].std() or 1.0
        stats[v] = {"mean": float(mu or 0.0), "std": float(sd or 1.0)}
        df = df.with_columns(((pl.col(ecol) - mu) / (sd + EPS)).alias(f"z_{ecol}"))
        err_cols.append(f"z_{ecol}")

    weights = cfg["targets"]["weights"]
    if weights:
        w = np.array([weights.get(v, 1.0) for v in tvars]); w = w / w.sum()
        df = df.with_columns(
            sum(w[i] * pl.col(err_cols[i]) for i in range(len(tvars))).alias("composite_error_z")
        )
    else:
        df = df.with_columns(pl.mean_horizontal(err_cols).alias("composite_error_z"))

    return df, tvars, stats


def bust_labels(df, cfg, trainval_mask_col, lead_col, zone_col):
    q = cfg["targets"]["default_quantile"]
    thresholds = (
        df.filter(pl.col(trainval_mask_col))
        .group_by([zone_col, lead_col])
        .agg(pl.col("composite_error_z").quantile(q).alias("thr"))
    )
    df = df.join(thresholds, on=[zone_col, lead_col], how="left")
    df = df.with_columns((pl.col("composite_error_z") > pl.col("thr")).cast(pl.Int8).alias("bust"))
    thr_table = thresholds.to_dicts()
    return df, thr_table, q


def assign_split(df, cfg, date_col):
    s = cfg["splits"]
    def in_range(col, lo, hi):
        return (pl.col(col) >= pl.lit(lo).str.strptime(pl.Date, "%Y-%m-%d")) & \
               (pl.col(col) <= pl.lit(hi).str.strptime(pl.Date, "%Y-%m-%d"))
    df = df.with_columns(
        pl.when(in_range(date_col, *s["train"])).then(pl.lit("train"))
        .when(in_range(date_col, *s["val"])).then(pl.lit("val"))
        .when(in_range(date_col, *s["calib"])).then(pl.lit("calib"))
        .when(in_range(date_col, *s["test"])).then(pl.lit("test"))
        .otherwise(pl.lit("unused")).alias("split")
    )
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gfs-daily", default=None)
    ap.add_argument("--era5-daily", default=None)
    ap.add_argument("--points", default=None)
    args = ap.parse_args()
    cfg = load_cfg()
    proc = Path(cfg["paths"]["processed"])
    feat_dir = ensure(P(cfg, "features"))
    c = cfg["columns"]

    points = pl.read_csv(args.points or P(cfg, "points"))
    gfs = pl.read_parquet(args.gfs_daily or (proc / "gfs_daily.parquet"))
    era5 = pl.read_parquet(args.era5_daily or (proc / "era5_daily.parquet"))

    print(f"gfs_daily: {gfs.shape}, era5_daily: {era5.shape}, points: {points.shape}")
    lead_col = c["lead"]
    max_lead = cfg["leads"]["target"][-1]
    gfs = gfs.filter(pl.col(lead_col).is_in(cfg["leads"]["target"]))
    gfs = gfs.with_columns((pl.col("date")).alias("valid_date"))

    # 1) spatial graph from the CANONICAL points file (not from either data source, since
    #    GFS/ERA5 coordinates are grid-cell interpolations that wobble slightly per source)
    print("Building spatial graph ...")
    graph_info = build_graph(points, cfg)
    print(" ", graph_info)
    graph_npz = np.load(feat_dir / "graph.npz")

    # 2) forecast-cycle revision (needs >=2 GFS cycles covering the same valid date; if the
    #    dataset only ever stores ONE cycle per valid date, this join naturally comes back
    #    all-null and script 05 will drop the *_cyclechange columns automatically)
    print("Computing forecast-cycle revision features ...")
    rev = gfs_revision(gfs, cfg)

    # 3) ERA5 temporal history (previous/rolling stats using data strictly before init_date)
    print("Computing ERA5 temporal-history features ...")
    hist = temporal_history(era5, cfg)

    # 4) spatial neighbourhood features from the most recent observed ERA5 state (t0-1)
    print("Computing spatial neighbourhood features (this can take a while: pure-python kNN loop) ...")
    spatial_vars = [v for v in cfg["features"]["spatial_vars"] if v in era5.columns]
    era5_for_spatial = era5.select([c["id"], "date"] + spatial_vars).rename({"date": "hist_asof_date"})
    spat = spatial_features(era5_for_spatial, cfg, graph_npz)

    # 5) assemble main sample table: one row per (point, init_date, lead)
    gfs_state_cols = [col for col in gfs.columns if col.endswith(("_mean", "_sum", "_max", "_min"))]
    main = gfs.select([c["id"], "init_date", lead_col, "valid_date"] + gfs_state_cols)
    main = main.rename({col: f"{col}_gfs" for col in gfs_state_cols})

    main = main.join(rev, on=[c["id"], "init_date", lead_col], how="left")
    main = main.with_columns((pl.col("init_date") - pl.duration(days=1)).alias("hist_asof_date"))
    main = main.join(hist, on=[c["id"], "hist_asof_date"], how="left")
    if spat is not None:
        main = main.join(spat, on=[c["id"], "hist_asof_date"], how="left")

    era5_truth_cols = [col for col in era5.columns if col.endswith(("_mean", "_sum", "_max", "_min"))]
    truth = era5.select([c["id"], "date"] + era5_truth_cols).rename(
        {col: f"{col}_era5" for col in era5_truth_cols})
    main = main.join(truth, left_on=[c["id"], "valid_date"], right_on=[c["id"], "date"], how="left")

    main = main.join(points.select(["id", "zone", "label", "latitude", "longitude"]),
                      left_on=c["id"], right_on="id", how="left")

    # 6) splits (by INIT date, gap between blocks > max_lead so no forecast crosses boundaries)
    main = assign_split(main, cfg, "init_date")
    n_drop = main.filter(pl.col("split") == "unused").height
    main = main.filter(pl.col("split") != "unused")
    print(f"Dropped {n_drop} rows outside configured train/val/calib/test windows.")

    # 7) targets: composite error + bust label (thresholds fit on train/train+val only)
    main = main.with_columns((pl.col("split") == "train").alias("_is_train"))
    main, tvars, err_stats = composite_error_and_bust(main, cfg, "_is_train")
    main = main.with_columns(pl.col("split").is_in(["train", "val"]).alias("_is_trainval"))
    main, thr_table, q_used = bust_labels(main, cfg, "_is_trainval", lead_col, "zone")
    main = main.drop(["_is_train", "_is_trainval"])

    print("Split sizes:", main.group_by("split").agg(pl.len()).to_dicts())
    print("Bust rate by split:", main.group_by("split").agg(pl.col("bust").mean()).to_dicts())

    out = feat_dir / "samples.parquet"
    main.write_parquet(out)
    print(f"Saved {main.height:,} rows x {main.width} cols -> {out}")

    # IMPORTANT: exclude anything derived from the truth (ERA5 valid-time values, raw/z-scored
    # per-variable errors, the composite error, the bust threshold) from the model's INPUT
    # features -- only *_gfs (forecast, known at init time), *_cyclechange, and the
    # ERA5-history/spatial columns computed strictly before init_date are legitimate inputs.
    excluded_exact = {c["id"], "init_date", lead_col, "valid_date", "hist_asof_date", "zone",
                      "label", "latitude", "longitude", "split", "bust", "composite_error_z", "thr"}
    excluded_prefixes = ("err_", "z_err_")
    excluded_suffixes = ("_era5",)
    feature_cols = [col for col in main.columns if col not in excluded_exact
                    and not col.startswith(excluded_prefixes)
                    and not col.endswith(excluded_suffixes)]
    leaked_check = [c for c in feature_cols if c.endswith("_era5") or "z_err_" in c or c.startswith("err_")]
    assert not leaked_check, f"Leakage guard tripped, would have leaked: {leaked_check}"
    meta = dict(graph=graph_info, error_vars=tvars, error_stats=err_stats, bust_quantile=q_used,
                bust_thresholds=thr_table, feature_columns=sorted(feature_cols),
                n_rows=main.height, leads=cfg["leads"]["target"])
    save_json(meta, feat_dir / "feature_meta.json")
    print(f"Saved feature metadata -> {feat_dir / 'feature_meta.json'}")


if __name__ == "__main__":
    main()
