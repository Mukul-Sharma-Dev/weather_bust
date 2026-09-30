# BustCast -- AI-Based Forecast Bust Detection for Medium-Range Weather Forecasts

Prototype for MoES / NCMRWF Problem Statement 26079. Given GFS forecasts and ERA5 reanalysis
over ~900 Indian points (monsoon seasons 2024-2025), predicts **where and at which lead time
(Day 1-7) the GFS forecast is likely to bust**, with calibrated confidence, error-prone-area
detection, and plain-language explanations.

This was built and unit-tested against your uploaded `india_points.csv` (900 real coordinates)
and small samples of your two data files. It has **not** been run against your actual
5.5 GB `forecast_gfs_2024_2025_list.csv` / `truth_era5_2024_2025_list.csv` -- I don't have
those files. Every script was validated end-to-end on synthetic data shaped exactly like your
real files (same columns, same `lead_day`/no-`lead_day` distinction, multi-year, 20-900 points)
so the pipeline runs; you'll want to watch script 01's output the first time you point it at
your real CSVs in case a column name differs from what's assumed.

## 1. Folder structure

```
forecast_bust/
├── config.yaml                  # every path, column name, split date, threshold, model
│                                 # hyperparameter -- read this first
├── requirements.txt
├── README.md                    # this file
│
├── bustcast/                    # importable package
│   ├── utils.py                 # config loading, seeding, GPU device reporting
│   ├── geo.py                   # grid-vs-graph detection, haversine kNN, gradient operator,
│   │                             # KMeans region pooling
│   ├── dataset.py                # long-parquet -> [N,F] / [N,L,F] tensors, hist/lead feature split
│   ├── model.py                  # SpatialEncoder (GAT) + TemporalEncoder (Transformer) +
│   │                             # CausalDecoder + multi-task heads
│   └── explain.py                # shared input-gradient saliency + factor-name mapping
│
├── scripts/
│   ├── 01_inspect_data.py        # schema/nulls/date-range/lead_day inspection (streaming)
│   ├── 02_build_daily.py         # hourly -> daily aggregation (streaming, polars)
│   ├── 03_build_features.py      # temporal/spatial/forecast-cycle features, targets, splits
│   ├── 04_train_baseline_xgb.py  # Stage 1: XGBoost baseline, per lead day
│   ├── 05_train_transformer.py   # Stage 2/3: main model, + --ablation A..E
│   ├── 06_calibrate.py           # Platt/isotonic probability calibration
│   ├── 07_evaluate.py            # lead-wise metrics, spatial maps, region/area detection
│   ├── 08_ablations.py           # runs all 5 ablations back to back
│   └── 09_explainability.py      # XGBoost permutation importance + Transformer saliency
│
├── app/
│   ├── api.py                    # FastAPI: POST /forecast -> spec Sec.27 JSON
│   └── dashboard.py               # Streamlit: Day1-7 selector, 4 maps, error-prone areas, explanations
│
├── data/
│   ├── raw/                      # put forecast_gfs_2024_2025_list.csv, truth_era5_2024_2025_list.csv,
│   │                             # and india_points.csv here
│   ├── processed/                # gfs_daily.parquet, era5_daily.parquet (script 02 output)
│   └── features/                 # samples.parquet, feature_meta.json, graph.npz (script 03 output)
│
└── outputs/
    ├── baselines/                 # XGBoost models + metrics
    ├── checkpoints/                # stage2_best.pt, stage3_best.pt, *_train_log.json
    ├── calibration/                # calibration_report.json, risk_bins.json
    ├── eval/                       # leadwise_metrics.csv, region_aggregation.csv,
    │                               # error_prone_areas.json, maps/lead{d}_maps.png
    ├── ablations/                  # ablation_results.json
    └── explainability/             # xgb_permutation_importance.json, sample_explanations.json
```

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# torch: the requirements.txt line installs whatever `pip install torch` resolves to for your
# platform. On the RTX 4060 laptop, that should already pull a CUDA build; if `python -c
# "import torch; print(torch.cuda.is_available())"` prints False, install from
# https://pytorch.org/get-started/locally/ with your CUDA version instead.
```

Put your three files under `data/raw/`:
```
data/raw/forecast_gfs_2024_2025_list.csv
data/raw/truth_era5_2024_2025_list.csv
data/raw/india_points.csv
```
(paths are configurable in `config.yaml` -> `paths:`)

## 3. Running the full pipeline

```bash
# 1. Look at the data before assuming anything about it
python scripts/01_inspect_data.py

# 2. Hourly -> daily (streams the 5.5GB CSVs; doesn't load them whole)
python scripts/02_build_daily.py

# 3. Feature engineering, targets, leakage-safe splits
python scripts/03_build_features.py

# 4. Baselines
python scripts/04_train_baseline_xgb.py

# 5. Main model -- staged
python scripts/05_train_transformer.py --stage 2                                    # regression only
python scripts/05_train_transformer.py --stage 3 --init_from outputs/checkpoints/stage2_best.pt

# 6. Calibrate P(bust) on the held-out calib split
python scripts/06_calibrate.py --checkpoint outputs/checkpoints/stage3_best.pt

# 7. Full evaluation: lead-wise table, spatial maps, region aggregation, error-prone areas
python scripts/07_evaluate.py --checkpoint outputs/checkpoints/stage3_best.pt

# 8. (optional, slower) which information sources actually help
python scripts/08_ablations.py --epochs 15

# 9. Explainability report
python scripts/09_explainability.py --checkpoint outputs/checkpoints/stage3_best.pt

# 10. Serve it
uvicorn app.api:app --host 0.0.0.0 --port 8000
streamlit run app/dashboard.py            # in another terminal; sidebar defaults to localhost:8000
```

Every script prints what it's doing and why (branch taken, thresholds chosen, etc.) rather than
silently assuming -- read the stdout the first time you run each one on your real files.

## 4. What script 01/03 actually found in *your* data

- `india_points.csv`: 900 points, 4 zones (`coastline`, `major_city`, `mountain_north`,
  `northeast`), latitudes 8.1-35.2N, longitudes 69.1-96.9E.
- **The 900 points are NOT a regular lat/lon grid** (`geo.detect_grid` returns a lattice-fill
  ratio of ~0.4%). So the spatial encoder uses **Case B: a k-nearest-neighbour graph +
  graph-attention**, not a CNN. `geo.choose_k` picked **k=8** for this geometry (median
  8th-neighbour distance ≈ 0.6° ≈ 67 km, comfortably inside the configured 1.0° budget).
- Your two sample rows show the GFS file carries a `lead_day` column and the ERA5 file does
  not -- exactly the reanalysis-vs.-forecast distinction the spec describes. Script 02/03 use
  that column's presence/absence to decide which pipeline branch a file takes automatically;
  nothing about lead-day count or column names is hard-coded beyond what `config.yaml` lists.
- **A caught bug worth knowing about**: an earlier version of script 03's feature-column filter
  let the ERA5 truth values and the per-variable error columns leak into the model's own input
  features (only excluding `err_` but not `z_err_`, and not excluding `*_era5` at all). That
  would have made every metric downstream meaningless. It's now fixed with an explicit
  allow-list plus an `assert` that fails loudly if it ever regresses -- see the "IMPORTANT"
  comment in `scripts/03_build_features.py` around `feature_cols`. If you fork this script,
  keep that assertion.

## 5. Architecture

```
                   GFS(lead 1..7) + ERA5 history (t0-1 .. t0-7)
                           |
                Feature Engineering (script 03)
                           |
            +--------------+--------------+
            |                             |
     ERA5 Temporal/Spatial          GFS per-lead state
     features (lead-independent)   + forecast-cycle revision
            |                       (lead-dependent)
            v                             |
     Spatial Encoder                      |
     (kNN Graph Attention, k=8)           |
            |                             |
     Temporal "views" (3 learned          |
     read-out heads) -> region            |
     pooling (64 KMeans regions)          |
            |                             |
     Temporal Transformer Encoder         |
     (geo + pseudo-time pos. embed.)      |
            |                             |
     7 causal lead-query tokens           |
     -> Causal Transformer Decoder        |
            |                             |
            +--------------+--------------+
                           v
              per-point concat: [lead context | node embedding | GFS-at-this-lead]
                           |
              +------------+------------+
              v                         v
       Huber Error Regression   Bust Probability (sigmoid)
        Error_hat[900,7]          P(bust)[900,7]
              |                         |
              +------------+------------+
                           v
                Probability Calibration (Platt/isotonic, fit on `calib` split)
                           |
                Confidence = 1 - calibrated P(bust)
```

**Design choices and why**, in the spec's own terms:

- **Spatial representation (Sec.6/7)**: irregular points -> graph attention over a kNN graph,
  auto-selected `k`, with a distance-weighted ridge-regression gradient operator
  (`geo.gradient_operator`) for the spatial-gradient features rather than blind finite
  differences, because the points aren't evenly spaced. Reported explicitly by script 03/01.
- **"Compressed spatial tokens" (Sec.25)**: rather than attending over all 900 points directly
  in the temporal Transformer, per-point GAT embeddings are pooled into 64 KMeans-clustered
  region tokens first, and the decoder's output is broadcast back to points (matching the
  spec's `900 -> compressed -> Temporal Transformer -> ... -> 900-location reconstruction`
  flow) via concatenation with each point's own GAT embedding, rather than literally
  un-pooling through the region matrix -- see the long comment in
  `model.py::decode_and_predict` for the exact reasoning.
- **Temporal axis (Sec.8)**: the dataset gives one engineered feature vector per (point,
  init_date), not a literal raw multi-day sequence, so the "previous atmospheric state ->
  current state" axis is built from 3 learned linear "read-out" views of the same GAT
  embedding (`TemporalViews` in `model.py`) rather than re-running the encoder 4 times on 4
  separate raw snapshots. If you later have a genuine per-day raw sequence to feed in, swap
  what's fed into `TemporalEncoder` -- nothing else needs to change.
- **Per-lead GFS conditioning (Sec.10)**: all 7 lead-day GFS fields are legitimately available
  at init time, so they're injected as a per-lead input at the decoder/head stage
  (`gfs_lead_proj`), not withheld by the causal mask.
- **Decoding strategy (Sec.11) -- reported explicitly, as the spec asks**: this implementation
  uses **PARALLEL, non-autoregressive decoding**. The 7 lead tokens are learned queries, not
  shifted teacher-forced error values, so there is no future-target information for the causal
  mask to leak in the first place; the mask is kept in the code so a literal
  teacher-forcing/autoregressive-inference variant can be dropped in later without touching the
  encoder. This is a real engineering trade-off (simpler, no inference-time autoregressive
  loop, no train/test mismatch risk) rather than an oversight -- flagging it because the spec
  explicitly wants this stated, not assumed.
- **Bust threshold (Sec.13)**: `percentile_threshold` per (zone, lead), fit on train+val only
  (`bust_labels` in script 03), default 90th percentile (configurable, 95th is also computed if
  you change `targets.default_quantile`).
- **Leakage guards**: composite-error z-normalisation stats fit on `train` only; bust
  thresholds fit on `train+val` only; calibration fit on the dedicated `calib` split;
  `test` is never touched by any of the above (script 03's `assign_split` + the
  `composite_error_and_bust`/`bust_labels` functions). See also the fixed leakage bug in
  section 4 above.

## 6. Metrics you'll get out of script 07

Per lead day (Day 1..7) and overall: MAE, RMSE, Pearson/Spearman r (regression);
ROC-AUC, PR-AUC, F1, Recall, False-Alarm-Rate, Miss-Rate, Brier (classification) --
`outputs/eval/leadwise_metrics.csv`. Six spatial maps per lead day (actual/predicted error,
actual/predicted bust, confidence, predicted-minus-actual) at real lat/lon coordinates --
`outputs/eval/maps/`. Region-wise aggregation and clustered error-prone areas (haversine-linked
connected components, not isolated single-cell "regions") -- `outputs/eval/region_aggregation.csv`
and `outputs/eval/error_prone_areas.json`.

## 7. Honesty notes / what to sanity-check first on the real data

- **The synthetic data used for testing had i.i.d. random forecast error with no real signal**,
  so ROC-AUC ~0.5 in the test runs you'd see if you replayed my test logs is *expected and
  correct for that synthetic data* -- it is not a claim about how the model will do on your
  real GFS/ERA5 files, where actual atmospheric error should be learnable from the engineered
  features.
- The `_cyclechange` (forecast-revision) features depend on your file actually storing more
  than one GFS cycle per valid date. Script 01 will tell you whether that's true; if your file
  only ever has one issue time per valid date, those columns will end up all-null, get dropped
  automatically in `dataset.py`'s median-fill path, and the model will simply not have that
  feature available -- it won't crash.
- Re-tune `config.yaml -> splits` once you see the real date range from script 01: the current
  windows are guesses roughly matching a 2024/2025 monsoon-season structure, with gaps between
  blocks bigger than the 7-day max lead (to stop any single forecast crossing a split boundary).
- `model.d_model`/`d_node`/layer counts in `config.yaml` are the spec's own "recommended initial
  configuration" (Sec.8/24). Scale up only if train/val loss says the model is underfitting --
  the spec explicitly warns against starting deep on two monsoon seasons of data.
