import argparse
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import CachedWindows, stats
from model import CausalTSF
from metrics import trajectory, summarize


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, mean, std, device, opt=None):
    train = opt is not None; model.train(train); total = 0.; count = 0
    for x, y, *_ in loader:
        x = (x.to(device) - mean) / std; y = y.to(device)
        pred = model(x); loss = torch.nn.functional.smooth_l1_loss(pred, y)
        if train:
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        total += float(loss.detach()) * len(x); count += len(x)
    return total / max(1, count)


@torch.no_grad()
def evaluate(model, ds, mean, std, device, batch):
    model.eval(); pred = []; target = []; rec = []
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    for x, y, *_ in loader:
        pred.append(model((x.to(device) - mean) / std).cpu().numpy()); target.append(y.numpy())
    pred, target = np.concatenate(pred), np.concatenate(target)
    for ri, idxs in ds.record_batches().items():
        p, y = pred[idxs], target[idxs]
        ate, rte = trajectory(p, y)
        rec.append({"record": ds.records[ri]["name"], "ate_xyz": ate, "rte30_xyz": rte,
                    "sq": float(np.sum((p-y)**2)), "n": len(p)})
    return summarize(rec), rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["time", "tsf", "tsf_patch", "tsf_gru"], default="time")
    ap.add_argument("--cache", default=str(Path(__file__).resolve().parents[2] / "cache" / "yt"))
    ap.add_argument("--out", default="runs/causal_tsf")
    ap.add_argument("--max-records", type=int, default=24)
    ap.add_argument("--max-windows", type=int, default=12000)
    ap.add_argument("--eval-max-records", type=int, default=None,
                    help="valid/test record cap; defaults to half the train cap")
    ap.add_argument("--eval-max-windows", type=int, default=None,
                    help="valid/test window cap; defaults to one third of train cap")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(); seed_all(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    cache = Path(args.cache).resolve(); out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    mean_np, std_np, meta = stats(cache); mean = torch.tensor(mean_np, device=device)[None,:,None]; std = torch.tensor(std_np, device=device)[None,:,None]
    train = CachedWindows(cache, "train", max_records=args.max_records, max_windows=args.max_windows)
    eval_records = args.eval_max_records if args.eval_max_records is not None else max(8, args.max_records // 2)
    eval_windows = args.eval_max_windows if args.eval_max_windows is not None else max(2000, args.max_windows // 3)
    valid = CachedWindows(cache, "valid", max_records=eval_records, max_windows=eval_windows)
    model = CausalTSF(args.variant, args.width).to(device)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-3, weight_decay=1e-4)
    loaders = {"train": DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=0),
               "valid": DataLoader(valid, batch_size=args.batch, shuffle=False, num_workers=0)}
    history = []
    for ep in range(1, args.epochs + 1):
        tr = run_epoch(model, loaders["train"], mean, std, device, opt)
        va = run_epoch(model, loaders["valid"], mean, std, device)
        history.append({"epoch": ep, "train_loss": tr, "valid_loss": va}); print(args.variant, ep, tr, va, flush=True)
    test = CachedWindows(cache, "test", max_records=eval_records, max_windows=eval_windows)
    val_metrics, _ = evaluate(model, valid, mean, std, device, args.batch)
    test_metrics, rows = evaluate(model, test, mean, std, device, args.batch)
    result = {"variant": args.variant, "seed": args.seed, "device": str(device), "parameters": model.count_parameters(),
              "records": {"train": len(train.records), "valid": len(valid.records), "test": len(test.records)},
              "windows": {"train": len(train), "valid": len(valid), "test": len(test)},
              "screen_caps": {"train_records": args.max_records, "train_windows": args.max_windows,
                              "eval_records": eval_records, "eval_windows": eval_windows},
              "valid": val_metrics, "test": test_metrics, "history": history,
              "protocol": "cached navigation frame; 1s input, 1s target, 1Hz non-overlap; full-record ATE and all legal RTE30 pairs",
              "cache_identity": meta["identity"]}
    (out / f"{args.variant}_seed{args.seed}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({"model": model.state_dict(), "variant": args.variant, "width": args.width, "mean": mean_np.tolist(), "std": std_np.tolist(), "cache_identity": meta["identity"]}, out / f"{args.variant}_seed{args.seed}.pt")
    print(json.dumps({"variant": args.variant, "valid": val_metrics, "test": test_metrics}, indent=2), flush=True)


if __name__ == "__main__": main()
