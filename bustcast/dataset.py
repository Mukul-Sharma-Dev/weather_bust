"""
Turns data/features/samples.parquet (long: point x init_date x lead) into the
[time, spatial_points, features] tensors the model consumes, one sample per
init_date. Missing (point, lead) combinations -- e.g. a point with no ERA5
truth yet for a very recent init -- are filled with zero + a validity mask
that both the loss and the metrics respect.
"""
from pathlib import Path
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from . import geo


HIST_SUFFIXES = ("_prev1", "_delta1", "_delta2", "_rollmean3", "_rollstd3", "_rollmean7",
                  "_rollstd7", "_localmean", "_localstd", "_anomaly", "_neighbdisagree")
LEAD_SUFFIXES = ("_gfs", "_cyclechange")


def split_feature_cols(feature_cols):
    """Lead-independent (ERA5 history / spatial) vs. lead-dependent (GFS state at that lead,
    forecast-cycle revision) feature groups -- see model.py's docstring for why they're fed
    to different parts of the network."""
    hist, lead = [], []
    for c in feature_cols:
        if c.endswith(LEAD_SUFFIXES):
            lead.append(c)
        elif c.endswith(HIST_SUFFIXES):
            hist.append(c)
        else:
            hist.append(c)  # safe default for anything uncategorised
    return hist, lead


class BustDataset(Dataset):
    def __init__(self, samples_path, graph_npz_path, feature_cols, error_var_cols,
                 split, leads, points_order=None, zone_list=None):
        df = pl.read_parquet(samples_path)
        df = df.filter(pl.col("split") == split)
        self.leads = leads
        self.feature_cols = feature_cols
        self.hist_cols, self.lead_cols = split_feature_cols(feature_cols)
        self.error_var_cols = error_var_cols  # z_err_<var> columns, per-variable regression targets

        g = np.load(graph_npz_path)
        self.point_id_order = points_order if points_order is not None else g["point_id"]
        self.n_points = len(self.point_id_order)
        self.coords = g["coords"]
        self.nbr, self.dist, self.grad_op, self.edge_feat = g["nbr"], g["dist"], g["grad_op"], g["edge_feat"]
        self.region_pool = g["region_pool"]
        pid_to_slot = {int(p): i for i, p in enumerate(self.point_id_order)}

        zones = zone_list or sorted(df["zone"].unique().to_list())
        self.zone_to_idx = {z: i for i, z in enumerate(zones)}

        # median-impute engineered features on TRAIN only would be ideal; for simplicity + because
        # scripts/03 already avoids inventing values, we impute per-column TRAIN median computed once
        # by the caller and passed in via `feature_medians` (see build_datasets below).
        self.df = df
        self._pid_to_slot = pid_to_slot
        self._index_by_init = df.select("init_date").unique().sort("init_date")["init_date"].to_list()

    def set_medians(self, medians: dict):
        self.medians = medians

    def __len__(self):
        return len(self._index_by_init)

    def __getitem__(self, idx):
        d = self._index_by_init[idx]
        day = self.df.filter(pl.col("init_date") == d)

        Xh = np.zeros((self.n_points, len(self.hist_cols)), dtype=np.float32)
        Xl = np.zeros((self.n_points, len(self.leads), len(self.lead_cols)), dtype=np.float32)
        Yerr = np.zeros((self.n_points, len(self.leads), len(self.error_var_cols)), dtype=np.float32)
        Ycomp = np.zeros((self.n_points, len(self.leads)), dtype=np.float32)
        Ybust = np.zeros((self.n_points, len(self.leads)), dtype=np.float32)
        mask = np.zeros((self.n_points, len(self.leads)), dtype=np.float32)
        zone_idx = np.zeros(self.n_points, dtype=np.int64)
        hist_done = np.zeros(self.n_points, dtype=bool)

        for row in day.iter_rows(named=True):
            slot = self._pid_to_slot.get(int(row["point_id"]))
            if slot is None:
                continue
            lead = row["lead_day"]
            if lead not in self.leads:
                continue
            li = self.leads.index(lead)
            if not hist_done[slot]:
                # hist/spatial columns are identical across lead rows for a given (point,init) --
                # fill once from whichever lead row is seen first for this point.
                for fi, fc in enumerate(self.hist_cols):
                    v = row.get(fc)
                    Xh[slot, fi] = v if v is not None else self.medians.get(fc, 0.0)
                zone_idx[slot] = self.zone_to_idx.get(row["zone"], 0)
                hist_done[slot] = True
            for fi, fc in enumerate(self.lead_cols):
                v = row.get(fc)
                Xl[slot, li, fi] = v if v is not None else self.medians.get(fc, 0.0)
            ce = row.get("composite_error_z")
            if ce is not None:
                Ycomp[slot, li] = ce
                mask[slot, li] = 1.0
            b = row.get("bust")
            if b is not None:
                Ybust[slot, li] = b
            for vi, ec in enumerate(self.error_var_cols):
                v = row.get(ec)
                if v is not None:
                    Yerr[slot, li, vi] = v

        return dict(
            Xh=torch.from_numpy(Xh), Xl=torch.from_numpy(Xl),
            Yerr=torch.from_numpy(Yerr), Ycomp=torch.from_numpy(Ycomp),
            Ybust=torch.from_numpy(Ybust), mask=torch.from_numpy(mask),
            zone_idx=torch.from_numpy(zone_idx), init_date=str(d),
        )


def collate(batch):
    out = {}
    for k in batch[0]:
        if k == "init_date":
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch], 0)
    return out


def compute_train_medians(samples_path, feature_cols):
    df = pl.read_parquet(samples_path).filter(pl.col("split") == "train")
    return {c: (df[c].median() or 0.0) for c in feature_cols if c in df.columns}


def build_datasets(samples_path, graph_npz_path, feature_cols, error_var_cols, leads):
    medians = compute_train_medians(samples_path, feature_cols)
    ds = {}
    zones = sorted(pl.read_parquet(samples_path)["zone"].unique().to_list())
    for split in ["train", "val", "calib", "test"]:
        d = BustDataset(samples_path, graph_npz_path, feature_cols, error_var_cols, split, leads,
                         zone_list=zones)
        d.set_medians(medians)
        ds[split] = d
    return ds, medians
