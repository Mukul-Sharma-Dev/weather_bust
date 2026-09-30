import json, random
from pathlib import Path
import numpy as np, yaml

ROOT = Path(__file__).resolve().parents[1]


def load_cfg(path=None):
    with open(path or ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def P(cfg, key):
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p


def ensure(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=_Enc)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def set_seed(s):
    random.seed(s); np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    except ImportError:
        pass


def get_device(verbose=True):
    import torch
    cuda = torch.cuda.is_available()
    dev = torch.device("cuda" if cuda else "cpu")
    if verbose:
        print("CUDA available      :", cuda)
        print("PyTorch CUDA version:", torch.version.cuda)
        if cuda:
            pr = torch.cuda.get_device_properties(0)
            print("GPU name            :", pr.name)
            print(f"GPU memory          : {pr.total_memory / 2**30:.1f} GiB")
    return dev
