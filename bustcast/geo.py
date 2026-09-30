"""Geometry: grid detection, kNN graph, local gradient operator, spatial regions."""
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

R_KM = 6371.0


def to_xyz(coords):
    la, lo = np.radians(coords[:, 0]), np.radians(coords[:, 1])
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1)


def haversine_km(c1, c2):
    la1, lo1, la2, lo2 = map(np.radians, (c1[..., 0], c1[..., 1], c2[..., 0], c2[..., 1]))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_KM * np.arcsin(np.sqrt(a))


def detect_grid(coords, tol=0.02, min_fill=0.8):
    """Regular lat/lon grid test: distinct lat/lon levels and fill ratio of the lat x lon lattice.
    On this dataset (900 scattered station-like points across 4 zones) this returns regular=False,
    which is why the pipeline builds a kNN graph (Case B) rather than a CNN grid (Case A)."""
    lat_l = np.unique(np.round(coords[:, 0] / tol) * tol)
    lon_l = np.unique(np.round(coords[:, 1] / tol) * tol)
    fill = len(coords) / (len(lat_l) * len(lon_l))
    info = dict(n_lat=len(lat_l), n_lon=len(lon_l), fill=float(fill), regular=bool(fill >= min_fill))
    if info["regular"]:
        info["rows"] = np.searchsorted(lat_l, np.round(coords[:, 0] / tol) * tol)
        info["cols"] = np.searchsorted(lon_l, np.round(coords[:, 1] / tol) * tol)
    return info


def knn(coords, k):
    xyz = to_xyz(coords)
    _, idx = cKDTree(xyz).query(xyz, k=k + 2)          # +2: guards against duplicate points
    nbr = np.empty((len(coords), k), dtype=np.int64)
    for i in range(len(coords)):
        cand = [j for j in idx[i] if j != i][:k]
        while len(cand) < k:                            # pad if fewer than k distinct neighbours exist
            cand.append(cand[-1] if cand else i)
        nbr[i] = cand
    dist = haversine_km(coords[:, None, :], coords[nbr])
    return nbr, dist


def choose_k(coords, gcfg):
    if gcfg["k"] != "auto":
        return int(gcfg["k"]), {}
    stats, best = {}, gcfg["k_candidates"][0]
    for k in gcfg["k_candidates"]:
        _, d = knn(coords, k)
        med_deg = float(np.median(d[:, -1]) / 111.0)
        stats[k] = med_deg
        if med_deg <= gcfg["median_dist_deg"]:
            best = k
    return min(best, gcfg["k_max"]), stats


def local_offsets(coords, nbr):
    la, lo = np.radians(coords[:, 0]), np.radians(coords[:, 1])
    north = R_KM * (la[nbr] - la[:, None])
    east = R_KM * np.cos(la)[:, None] * (lo[nbr] - lo[:, None])
    return east, north


def gradient_operator(east, north, ridge=0.05, scale=100.0):
    """Distance-weighted ridge least-squares plane fit -> operator G [N,2,K]: grad = G @ (x_nbr - x_i).
    Units: per 100 km. Works for irregular points; ridge shrinks gradients where neighbours are
    nearly co-located or nearly collinear (common at coastline/mountain stations)."""
    A = np.stack([east, north], -1) / scale
    w = 1.0 / ((A ** 2).sum(-1) + 0.05)
    AtWA = np.einsum("nk,nki,nkj->nij", w, A, A) + ridge * np.eye(2)
    AtW = np.einsum("nki,nk->nik", A, w)
    return np.linalg.solve(AtWA, AtW).astype(np.float32)


def edge_features(east, north, dist):
    return np.stack([east / 100, north / 100, dist / 100], -1).astype(np.float32)


def make_regions(coords, n_regions, seed=0):
    n_regions = int(min(n_regions, max(2, len(coords) // 4)))
    lab = KMeans(n_regions, n_init=4, random_state=seed).fit(to_xyz(coords)).labels_
    cent = np.stack([coords[lab == m].mean(0) for m in range(n_regions)])
    pool = np.zeros((n_regions, len(coords)), np.float32)
    pool[lab, np.arange(len(coords))] = 1.0
    pool /= pool.sum(1, keepdims=True)
    return lab.astype(np.int64), cent.astype(np.float32), pool


def norm_coords(coords):
    """lat/lon -> [-1, 1] (fixed India-wide box; independent of the sample)."""
    lat = (coords[..., 0] - 20.0) / 20.0
    lon = (coords[..., 1] - 82.0) / 20.0
    return np.stack([lat, lon], -1).astype(np.float32)
