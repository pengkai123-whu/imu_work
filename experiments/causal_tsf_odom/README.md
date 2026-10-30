# Causal TSF Odometry Experiments

This directory implements the incremental experiment plan in `research_review/03_CAUSAL_METHOD/README.md`.

Variants:

- `time`: small time-domain CNN baseline.
- `tsf`: adds signed Fourier real/imag features, two-node accelerometer/gyroscope graph mixing, and signed Haar low/high features.
- `tsf_patch`: adds a lightweight cross-patch mixer inspired by recent ICDE forecasting work.
- `tsf_gru`: adds a one-way order-aware window mixer inspired by hierarchical/autoregressive forecasting, while keeping the same causal block contract. It is retained as a negative control only when validation is worse than `tsf_patch`.

All variants use the existing train-only navigation cache, 200-sample input, 200-sample displacement target, non-overlapping 1 Hz updates, and the same ATE/RTE30/step-RMSE code. The first pass is deliberately bounded by `--max-records` and `--max-windows`; it is a screening experiment, not the final full-data result.

## Fast screening

```powershell
conda activate spk
cd C:\Users\19510\Desktop\work_space\imu_work\meio_yt\experiments\causal_tsf_odom
python train.py --variant time --out ..\..\runs\screen_time --epochs 8
python train.py --variant tsf --out ..\..\runs\screen_tsf --epochs 8
python train.py --variant tsf_patch --out ..\..\runs\screen_tsf_patch --epochs 8
python train.py --variant tsf_gru --out ..\..\runs\screen_tsf_gru --epochs 8
```

For a fair screen, use the same record/window caps for every candidate and for `baseline.py`:

```powershell
python train.py --variant tsf_patch --out ..\..\runs\screen_tsf_patch20_fair --epochs 20 --max-records 24 --max-windows 12000 --eval-max-records 24 --eval-max-windows 12000 --batch 1024 --device cuda
python baseline.py --checkpoint ..\..\runs\yt\best.pt --out results\screen_meio.json --max-records 24 --max-windows 12000 --batch 1024 --device cuda
python record_screening.py
```

The screen uses 24 records and bounded windows. Validation selects modules; the test values in `results/screening.json` are a frozen diagnostic and are not used to tune thresholds. The retained `tsf_patch` candidate is then rerun with `--max-records 0 --max-windows 0 --eval-max-records 0 --eval-max-windows 0` for the full-data comparison. `results/full_comparison.json` records that comparison against the existing MEIO report. JSON experiment records are committed; weights and runs are ignored by the parent repository.

## Screening record

The screen was run with seed 6, 20 epochs, CUDA, and the same cached navigation-frame normalization. Results in meters:

| variant | full XYZ ATE | RTE30 | one-step displacement RMSE |
|---|---:|---:|---:|
| frozen MEIO checkpoint | 1.8849 | 1.2034 | 0.0941 |
| time | 8.0175 | 5.2130 | 0.2537 |
| TSF | 7.3846 | 4.8150 | 0.2405 |
| TSF + cross-patch | 7.3630 | 4.4548 | 0.2192 |
| TSF + window GRU | 7.8036 | 4.5206 | 0.2433 |

The TSF branches improve the small-model time baseline, and cross-patch is the best of these candidates on the validation selection rule. The frozen MEIO checkpoint was trained on all training records, so it is a stronger data-budget baseline than the 24-record screen; this table is not a final claim.

The full `tsf_patch` run (20 epochs, all records) reaches `ATE=3.1991`, `RTE30=1.5297`, and `DispRMSE=0.1242`, compared with the existing MEIO final report `1.8550`, `1.0589`, and `0.09745`. The current candidate therefore does not meet the requested 30% improvement threshold. It is a reproducible candidate implementation and ablation scaffold, not yet a publishable breakthrough.
