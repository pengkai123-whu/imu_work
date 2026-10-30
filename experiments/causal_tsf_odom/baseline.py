"""Evaluate the frozen MEIO-YT checkpoint on the causal screening subset.

This is intentionally an evaluator, not a second training implementation. It
uses the same CachedWindows and metrics as the candidate models so comparisons
are made on identical records, windows, and trajectory reconstruction rules.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from meio.model import MEIO
from meio.common import load_checkpoint
from data import CachedWindows
from metrics import trajectory, summarize


@torch.no_grad()
def evaluate(model, ds, mean, std, device, batch):
    model.eval()
    predictions, targets = [], []
    for x, y, *_ in DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0):
        x = (x.to(device) - mean) / std
        pred, _ = model.final(x)
        predictions.append(pred.cpu().numpy())
        targets.append(y.numpy())
    pred = np.concatenate(predictions)
    target = np.concatenate(targets)
    rows = []
    for ri, indices in ds.record_batches().items():
        p, y = pred[indices], target[indices]
        ate, rte = trajectory(p, y)
        rows.append({
            "record": ds.records[ri]["name"],
            "ate_xyz": ate,
            "rte30_xyz": rte,
            "sq": float(np.sum((p - y) ** 2)),
            "n": len(p),
        })
    return summarize(rows), rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="../../runs/yt/best.pt")
    parser.add_argument("--cache", default=str(Path(__file__).resolve().parents[2] / "cache" / "yt"))
    parser.add_argument("--out", default="results/screen_meio.json")
    parser.add_argument("--max-records", type=int, default=24)
    parser.add_argument("--max-windows", type=int, default=12000)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    checkpoint = Path(args.checkpoint).resolve()
    saved = load_checkpoint(checkpoint)
    model = MEIO(saved["model_cfg"]).to(device)
    model.load_state_dict(saved["model"])
    mean = torch.tensor(saved["mean"], dtype=torch.float32, device=device)[None, :, None]
    std = torch.tensor(saved["std"], dtype=torch.float32, device=device)[None, :, None]
    cache = Path(args.cache).resolve()
    test = CachedWindows(cache, "test", max_records=args.max_records, max_windows=args.max_windows)
    metrics, records = evaluate(model, test, mean, std, device, args.batch)
    result = {
        "variant": "meio_frozen_checkpoint",
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": saved.get("epoch"),
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "records": len(test.records),
        "windows": len(test),
        "test": metrics,
        "per_record": records,
        "protocol": "same CachedWindows subset; 1s input, 1s target, 1Hz non-overlap; full-record ATE and all legal RTE30 pairs",
    }
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["test"], indent=2), flush=True)


if __name__ == "__main__":
    main()
