"""
Inspect the raw GFS / ERA5 CSVs before any modelling decision is made.
Prints: schema, row/point counts, timestamp coverage, null rates, whether
`lead_day` / multiple init cycles exist for the same valid time, and whether
GFS and ERA5 share identical point_ids. Nothing here is written back to disk;
its only job is to tell scripts 02/03 what dynamic branches to take.

Usage:
    python scripts/01_inspect_data.py --gfs data/raw/forecast_gfs_2024_2025_list.csv \
                                       --era5 data/raw/truth_era5_2024_2025_list.csv
"""
import argparse, sys
from pathlib import Path
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bustcast.utils import load_cfg, P, save_json, ensure


def schema_report(path, name):
    print(f"\n=== {name}: {path} ===")
    lf = pl.scan_csv(path, try_parse_dates=False, infer_schema_length=5000)
    schema = lf.collect_schema()
    print("columns:", list(schema.names()))
    n_rows = lf.select(pl.len()).collect(engine="streaming").item()
    print("rows:", n_rows)
    null_frac = (
        lf.select([pl.col(c).null_count().alias(c) for c in schema.names()])
        .collect(engine="streaming")
    )
    tot = null_frac.to_dicts()[0]
    print("null fraction per column:")
    for c, v in tot.items():
        if v:
            print(f"  {c}: {v / n_rows:.4%}")
    return schema, n_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gfs", default=None)
    ap.add_argument("--era5", default=None)
    ap.add_argument("--points", default=None)
    args = ap.parse_args()

    cfg = load_cfg()
    gfs_path = args.gfs or P(cfg, "raw_gfs")
    era5_path = args.era5 or P(cfg, "raw_era5")
    pts_path = args.points or P(cfg, "points")

    pts = pl.read_csv(pts_path)
    print(f"points file: {pts_path} -> {pts.height} coordinates, zones={pts['zone'].unique().to_list()}")

    gfs_schema, gfs_rows = schema_report(gfs_path, "GFS forecast")
    era5_schema, era5_rows = schema_report(era5_path, "ERA5 reference")

    c = cfg["columns"]
    report = {"gfs_columns": list(gfs_schema.names()), "era5_columns": list(era5_schema.names()),
              "gfs_rows": gfs_rows, "era5_rows": era5_rows}

    has_lead_gfs = c["lead"] in gfs_schema.names()
    has_lead_era5 = c["lead"] in era5_schema.names()
    report["gfs_has_lead_day"] = has_lead_gfs
    report["era5_has_lead_day"] = has_lead_era5
    print(f"\nGFS has '{c['lead']}' column : {has_lead_gfs}")
    print(f"ERA5 has '{c['lead']}' column: {has_lead_era5}  "
          f"(expected False -- ERA5 is reanalysis, not a lead-dependent forecast)")

    # point_id coverage overlap
    gfs_ids = set(pl.scan_csv(gfs_path).select(c["id"]).unique().collect(engine="streaming")[c["id"]].to_list())
    era5_ids = set(pl.scan_csv(era5_path).select(c["id"]).unique().collect(engine="streaming")[c["id"]].to_list())
    pts_ids = set(pts["id"].to_list())
    report["n_points_gfs"] = len(gfs_ids)
    report["n_points_era5"] = len(era5_ids)
    report["n_points_master"] = len(pts_ids)
    report["gfs_era5_id_overlap"] = len(gfs_ids & era5_ids)
    print(f"\npoint_id overlap: master={len(pts_ids)} gfs={len(gfs_ids)} era5={len(era5_ids)} "
          f"common(gfs,era5)={len(gfs_ids & era5_ids)}")

    # date coverage + whether multiple GFS cycles exist for the same (point, lead, valid date)
    for path, schema, tag in [(gfs_path, gfs_schema, "gfs"), (era5_path, era5_schema, "era5")]:
        dt = pl.scan_csv(path, try_parse_dates=False).select(
            pl.col(c["time"]).str.slice(0, 10).alias("d")
        ).collect(engine="streaming")["d"]
        report[f"{tag}_date_min"], report[f"{tag}_date_max"] = str(dt.min()), str(dt.max())
        print(f"{tag} date range: {dt.min()} .. {dt.max()}")

    if has_lead_gfs:
        leads = pl.scan_csv(gfs_path).select(c["lead"]).unique().collect(engine="streaming")[c["lead"]].sort().to_list()
        report["gfs_lead_values"] = leads
        print("GFS lead_day values found:", leads)
        dup = (
            pl.scan_csv(gfs_path)
            .group_by([c["id"], c["time"], c["lead"]]).agg(pl.len().alias("n"))
            .filter(pl.col("n") > 1)
            .collect(engine="streaming")
        )
        report["gfs_duplicate_point_time_lead_rows"] = dup.height
        print(f"duplicate (point,datetime,lead) rows in GFS (would mean >1 init cycle stored): {dup.height}")

    out = ensure(P(cfg, "processed")) / "data_inspection_report.json"
    save_json(report, out)
    print(f"\nSaved machine-readable report -> {out}")


if __name__ == "__main__":
    main()
