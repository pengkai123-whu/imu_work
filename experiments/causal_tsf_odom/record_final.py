"""Record the full-data candidate against the existing formal MEIO report."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    baseline = json.loads((ROOT / "runs" / "yt" / "summary.json").read_text(encoding="utf-8"))
    candidate = json.loads((ROOT / "runs" / "full_tsf_patch20" / "tsf_patch_seed6.json").read_text(encoding="utf-8"))
    base = {
        "ate_xyz": baseline["final"]["macro"]["ate_xyz"],
        "rte30_xyz": baseline["final"]["macro"]["rte30_xyz"],
        "displacement_rmse_xyz": baseline["final"]["displacement_rmse"],
    }
    cand = candidate["test"]
    delta = {key: 1.0 - cand[key] / base[key] for key in base}
    result = {
        "protocol": {
            "candidate": "full train/test cache; 1s input, 1s target, 1Hz non-overlap",
            "baseline": "existing runs/yt frozen final MEIO report",
            "ate": "full record, initial translation only, record-macro",
            "rte": "all legal 30-second endpoint pairs, record-macro",
            "displacement": "pooled XYZ one-step RMSE",
            "candidate_epochs": 20,
            "candidate_seed": 6,
        },
        "baseline_meio": base,
        "candidate_tsf_patch": cand,
        "relative_reduction_candidate_vs_baseline": delta,
        "thirty_percent_breakthrough": all(v >= 0.30 for v in delta.values()),
        "interpretation": "The retained candidate is not yet better than MEIO on the full test split; do not claim a 30% breakthrough.",
    }
    output = HERE / "results" / "full_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
