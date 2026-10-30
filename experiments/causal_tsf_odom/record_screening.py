"""Create a versioned, machine-readable comparison table for screening runs."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
OUT = Path(__file__).resolve().parent / "results" / "screening.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(path, key="test"):
    payload = read(path)
    return {k: payload[key][k] for k in ("ate_xyz", "rte30_xyz", "displacement_rmse_xyz")}


def relative(candidate, reference):
    return {k: 1.0 - candidate[k] / reference[k] for k in reference}


def main():
    runs = {
        "meio_frozen": Path(__file__).resolve().parent / "results" / "screen_meio.json",
        "time": RUNS / "screen_time20_fair" / "time_seed6.json",
        "tsf": RUNS / "screen_tsf20_fair" / "tsf_seed6.json",
        "tsf_patch": RUNS / "screen_tsf_patch20_fair" / "tsf_patch_seed6.json",
        "tsf_gru": RUNS / "screen_tsf_gru20_fair" / "tsf_gru_seed6.json",
    }
    table = {}
    for name, path in runs.items():
        if not path.exists():
            raise FileNotFoundError(path)
        table[name] = metrics(path)
    reference = table["meio_frozen"]
    result = {
        "protocol": {
            "cache": "cache/yt",
            "train_records": 24,
            "train_window_cap": 12000,
            "eval_records": 24,
            "eval_window_cap": 12000,
            "input_seconds": 1.0,
            "target_seconds": 1.0,
            "update_seconds": 1.0,
            "ate": "full record, initial translation only, record-macro",
            "rte": "all legal 30-second endpoint pairs, record-macro",
            "displacement": "pooled XYZ one-step RMSE",
        },
        "metrics_m": table,
        "relative_reduction_vs_frozen_meio": {
            name: relative(values, reference)
            for name, values in table.items()
            if name != "meio_frozen"
        },
        "selection": {
            "retained": "tsf_patch",
            "rejected": {
                "tsf_gru": "validation metrics worse than tsf_patch",
            },
            "status": "screening only; frozen MEIO used a full-data checkpoint, so this is not a final train-budget comparison",
            "thirty_percent_breakthrough": False,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
