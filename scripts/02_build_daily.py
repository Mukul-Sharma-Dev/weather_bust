"""
Hourly -> daily aggregation for the GFS forecast file and the ERA5 reference
file. Runs fully lazily with polars' streaming engine so the ~5.5 GB CSVs
never have to fit in RAM at once.

For GFS: groups by (point_id, lead_day, valid_date). Also derives
`init_date = valid_date - lead_day days`, i.e. the forecast-cycle issue date,
which is what lets us build the forecast-revision features in script 03
(GFS(t0) - GFS(t-24h) for the same valid date/point, issued a day apart).

For ERA5: groups by (point_id, date). No lead_day, because a reanalysis has
no forecast horizon -- this matches what script 01 reports.

Wind speed/direction are first converted to u/v components (so averaging
across the day doesn't do a circular-mean-of-degrees mistake), then u/v are
aggregated along with everything else.

Output: data/processed/gfs_daily.parquet, data/processed/era5_daily.parquet
"""
import argparse, sys
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bustcast.utils import load_cfg, P, ensure


def add_uv(lf, spd_col, dir_col, prefix):
    """Meteorological convention: wind FROM direction dir_col (deg), speed spd_col."""
    rad = pl.col(dir_col).cast(pl.Float64) * (np.pi / 180.0)
    return lf.with_columns([
        (-pl.col(spd_col) * rad.sin()).alias(f"{prefix}_u"),
        (-pl.col(spd_col) * rad.cos()).alias(f"{prefix}_v"),
    ])


def build_agg_exprs(cfg, available_cols):
    aggs = []
    for var, stats in cfg["daily"]["agg"].items():
        if var not in available_cols:
            continue
        for stat in stats:
            e = getattr(pl.col(var), stat)()
            aggs.append(e.alias(f"{var}_{stat}"))
    for prefix in ("wind10", "wind100"):
        for comp in ("u", "v"):
            col = f"{prefix}_{comp}"
            if col in available_cols:
                aggs.append(pl.col(col).mean().alias(f"{col}_mean"))
    aggs.append(pl.col(cfg["columns"]["time"]).count().alias("n_hours"))
    return aggs


def process(path, cfg, is_gfs, out_path):
    c = cfg["columns"]
    lf = pl.scan_csv(path, try_parse_dates=False)
    schema_cols = set(lf.collect_schema().names())

    lf = lf.with_columns(pl.col(c["time"]).str.strptime(pl.Datetime, strict=False).alias("_dt"))
    lf = lf.with_columns(pl.col("_dt").dt.date().alias("_date"))

    # numeric columns can arrive as Utf8 if a source is entirely blank (e.g. no rain in this
    # sample) -- cast every configured/aggregatable variable to Float64 so sum/mean don't choke.
    numeric_candidates = set(cfg["daily"]["agg"].keys()) | {
        "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_direction_100m"
    }
    lf = lf.with_columns([
        pl.col(v).cast(pl.Float64, strict=False) for v in numeric_candidates if v in schema_cols
    ])

    if "wind_speed_10m" in schema_cols and "wind_direction_10m" in schema_cols:
        lf = add_uv(lf, "wind_speed_10m", "wind_direction_10m", "wind10")
    if "wind_speed_100m" in schema_cols and "wind_direction_100m" in schema_cols:
        lf = add_uv(lf, "wind_speed_100m", "wind_direction_100m", "wind100")

    available = schema_cols | {"wind10_u", "wind10_v", "wind100_u", "wind100_v"}
    aggs = build_agg_exprs(cfg, available)

    group_cols = [c["id"], "_date"]
    if is_gfs and c["lead"] in schema_cols:
        group_cols.append(c["lead"])

    daily = (
        lf.group_by(group_cols)
        .agg(aggs)
        .rename({"_date": "date"})
    )
    if is_gfs and c["lead"] in schema_cols:
        daily = daily.with_columns(
            (pl.col("date") - pl.duration(days=pl.col(c["lead"]))).alias("init_date")
        )

    # drop columns that ended up entirely null (e.g. rain/snowfall absent for this source)
    daily = daily.collect(engine="streaming")
    null_all = [c2 for c2 in daily.columns if daily[c2].null_count() == daily.height]
    if null_all:
        print(f"  dropping all-null columns: {null_all}")
        daily = daily.drop(null_all)

    daily.write_parquet(out_path)
    print(f"  -> {out_path}  rows={daily.height:,}  cols={daily.width}")
    return daily


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gfs", default=None)
    ap.add_argument("--era5", default=None)
    args = ap.parse_args()
    cfg = load_cfg()
    out_dir = ensure(P(cfg, "processed"))

    gfs_path = args.gfs or P(cfg, "raw_gfs")
    era5_path = args.era5 or P(cfg, "raw_era5")

    print("Aggregating GFS ->", gfs_path)
    process(gfs_path, cfg, is_gfs=True, out_path=out_dir / "gfs_daily.parquet")

    print("Aggregating ERA5 ->", era5_path)
    process(era5_path, cfg, is_gfs=False, out_path=out_dir / "era5_daily.parquet")


if __name__ == "__main__":
    main()
