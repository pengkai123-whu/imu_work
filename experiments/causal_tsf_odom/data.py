import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class CachedWindows(Dataset):
    """Read the existing train-only normalized navigation cache without copying it."""

    def __init__(self, cache, split, window=200, horizon=200, stride=200,
                 max_records=None, max_windows=None):
        self.cache = Path(cache)
        self.meta = json.loads((self.cache / "manifest.json").read_text(encoding="utf-8"))
        self.signals = np.load(self.cache / "signals.npy", mmap_mode="r")
        self.positions = np.load(self.cache / "positions.npy", mmap_mode="r")
        self.window, self.horizon, self.stride = int(window), int(horizon), int(stride)
        rows = [r for r in self.meta["records"] if r["split"] == split]
        if max_records:
            rows = rows[:int(max_records)]
        self.records = []
        for r in rows:
            n = max(0, (int(r["n"]) - self.window - 1) // self.stride + 1)
            if n:
                self.records.append({**r, "count": n})
        self.index = []
        for ri, r in enumerate(self.records):
            for j in range(r["count"]):
                self.index.append((ri, j * self.stride))
        if max_windows and len(self.index) > int(max_windows):
            self.index = self.index[:int(max_windows)]
        if not self.index:
            raise ValueError(f"no windows in {split}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ri, rel = self.index[i]
        r = self.records[ri]
        start = int(r["offset"]) + rel
        x = np.asarray(self.signals[start:start + self.window], dtype=np.float32).T
        y = np.asarray(self.positions[start + self.horizon] - self.positions[start], dtype=np.float32)
        # Cached arrays are memory-mapped read-only; copy the small window so
        # PyTorch never receives a non-writable NumPy view.
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy()), ri, rel

    def record_batches(self):
        """Return indices grouped by record for trajectory metrics."""
        groups = {i: [] for i in range(len(self.records))}
        for idx, (ri, _) in enumerate(self.index):
            groups[ri].append(idx)
        return groups


def stats(cache):
    meta = json.loads((Path(cache) / "manifest.json").read_text(encoding="utf-8"))
    return (np.asarray(meta["mean"], dtype=np.float32),
            np.asarray(meta["std"], dtype=np.float32), meta)
