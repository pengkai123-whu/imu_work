"""Evaluate a MEIO-family checkpoint on the same bounded screen subset."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from meio.common import load_checkpoint
from meio.model import make_model
from data import CachedWindows
from metrics import trajectory, summarize


@torch.no_grad()
def evaluate(model, ds, mean, std, device, batch):
    model.eval(); predictions=[]; targets=[]
    for x, y, *_ in DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0):
        means, _ = model((x.to(device)-mean)/std)
        predictions.append(means[:, -1].cpu().numpy()); targets.append(y.numpy())
    pred, target = np.concatenate(predictions), np.concatenate(targets)
    rows=[]
    for ri, indices in ds.record_batches().items():
        p, y = pred[indices], target[indices]
        ate, rte = trajectory(p, y)
        rows.append({"record": ds.records[ri]["name"], "ate_xyz": ate,
                     "rte30_xyz": rte, "sq": float(np.sum((p-y)**2)), "n": len(p)})
    return summarize(rows), rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache", default=str(ROOT/"cache"/"yt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-records", type=int, default=24)
    ap.add_argument("--max-windows", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--device", default="auto")
    a=ap.parse_args()
    device=torch.device("cuda" if a.device=="auto" and torch.cuda.is_available() else a.device)
    saved=load_checkpoint(a.checkpoint, device="cpu")
    model=make_model(saved["model_cfg"]).to(device); model.load_state_dict(saved["model"])
    mean=torch.tensor(saved["mean"],device=device,dtype=torch.float32)[None,:,None]
    std=torch.tensor(saved["std"],device=device,dtype=torch.float32)[None,:,None]
    ds=CachedWindows(a.cache,"test",max_records=a.max_records,max_windows=a.max_windows)
    metrics, rows=evaluate(model,ds,mean,std,device,a.batch)
    result={"variant":saved["model_cfg"].get("variant","meio"),"checkpoint":str(Path(a.checkpoint).resolve()),
            "records":len(ds.records),"windows":len(ds),"test":metrics,"per_record":rows,
            "protocol":"same CachedWindows subset; final MEIO exit; full-record ATE and all legal RTE30 pairs"}
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(metrics,indent=2))


if __name__ == "__main__": main()
