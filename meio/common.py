import hashlib
import json
import math
import random
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

def read_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    for section, key in [("data", "h5"), ("data", "cache"), ("train", "out")]:
        cfg[section][key] = str((path.parent / cfg[section][key]).resolve())
    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    if d["window"] != m["window"] or d["window"] < 1 or d["horizon"] < 1:
        raise ValueError("positive, matching input windows and positive horizon required")
    if m["patch"] < 1 or d["window"] % m["patch"] or m["depth"] < 1 or m["dim"] < 1 or m["expansion"] <= 0:
        raise ValueError("window must be divisible by patch; positive dim/depth required")
    if d["fs"] <= 0 or min(d["train_stride"], d["valid_stride"]) < 1:
        raise ValueError("fs and strides must be positive")
    if not isinstance(d.get('eval_stride',d['horizon']),int) or not 0 < d.get('eval_stride',d['horizon']) <= d['horizon']:
        raise ValueError('eval_stride must be a positive integer <= horizon')
    if t["mse_epochs"] < 1 or t["nll_epochs"] < 1 or t["transition_epochs"] < 0:
        raise ValueError("require positive MSE/NLL phases and nonnegative transition")
    if t["batch"] < 1 or t["workers"] < 0 or t["lr"] <= 0 or t["gamma"] < 0:
        raise ValueError("invalid optimizer/loader configuration")
    if not 0 < m["min_sigma"] < m["max_sigma"] or not 0 <= m["dropout"] < 1:
        raise ValueError("invalid sigma limits or dropout")
    if m.get('architecture','legacy') not in ['legacy','paper_fig3']:
        raise ValueError('unknown architecture')
    if t.get('supervision','all') not in ['all','final']:
        raise ValueError('supervision must be all or final')
    for key in ["max_records", "max_samples"]:
        if d.get(key) is not None and d[key] < 1:
            raise ValueError(key + " must be positive or null")
    return cfg

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)

def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def device_for(name):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

def load_checkpoint(path, device="cpu"):
    # Load only checkpoints created by this project / trusted by the user.
    return torch.load(path, map_location=device, weights_only=False)

def save_checkpoint(path, value):
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary); temporary.replace(path)

def checked_tensor(value, label):
    if not torch.isfinite(value).all():
        raise FloatingPointError("non-finite " + label)
