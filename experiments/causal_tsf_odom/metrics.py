import math
import numpy as np


def trajectory(pred, target, seconds=1.0):
    pred, target = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    est = np.vstack((np.zeros((1, 3)), np.cumsum(pred, axis=0)))
    gt = np.vstack((np.zeros((1, 3)), np.cumsum(target, axis=0)))
    err = est - gt
    lag = int(round(30 / seconds))
    ate = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))
    rte = float(np.sqrt(np.mean(np.sum((err[lag:] - err[:-lag]) ** 2, axis=1)))) if len(err) > lag else None
    return ate, rte


def summarize(rows):
    return {
        "ate_xyz": float(np.mean([r["ate_xyz"] for r in rows])),
        "rte30_xyz": float(np.mean([r["rte30_xyz"] for r in rows if r["rte30_xyz"] is not None])) if any(r["rte30_xyz"] is not None for r in rows) else None,
        "displacement_rmse_xyz": float(np.sqrt(sum(r["sq"] for r in rows) / sum(r["n"] for r in rows))),
        "records": len(rows), "windows": sum(r["n"] for r in rows)
    }
