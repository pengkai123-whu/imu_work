# Incremental MEIO + TSF Screening Log

All rows use the same 24-record screening subset, the final MEIO exit, full-record XYZ ATE, all legal RTE30 pairs, and pooled one-step displacement RMSE. The subset is for module selection only; it is not a final full-data claim.

| variant | ATE XYZ (m) | RTE30 XYZ (m) | step RMSE (m) | decision |
|---|---:|---:|---:|---|
| frozen MEIO | 1.88494 | 1.20338 | 0.09409 | reference |
| three-domain residual, joint fine-tuning | **1.61655** | **1.00785** | 0.08973 | keep |
| Fourier-only residual | 1.61660 | 1.00775 | 0.08973 | reject as no measurable advantage |
| graph-only residual | 1.61698 | 1.00795 | 0.08973 | reject |
| wavelet-only residual | 1.61714 | 1.00809 | 0.08973 | reject |
| frozen MEIO + three-domain residual | 1.76183 | 1.13404 | 0.09304 | diagnostic only |
| three-domain + causal patch mixer | 1.65850 | 1.03243 | 0.08947 | reject |
| three-domain + input diagonal SSM residual | 1.65849 | 1.03252 | 0.08947 | reject |
| three-domain + output residual | 1.65821 | 1.03228 | 0.08947 | reject |
| three-domain, 50-epoch budget | 1.78359 | 1.04026 | 0.08965 | reject |
| direct block SSM replacement | 24.03796 | 14.66712 | 0.59352 | reject; unstable |
| MLP + gated SSM residual in all blocks | 1.65564 | 1.02411 | **0.08883** | reject; trajectory metrics worse |
| MLP + gated SSM residual in blocks 6--7 | 1.65457 | 1.02383 | 0.08884 | reject |
| three-domain + local trend consistency loss | 1.64985 | 1.02813 | 0.08928 | reject |
| three-domain + sample-level causal SSM | 1.65475 | 1.02364 | 0.08885 | reject |
| three-domain + multi-scale temporal residual | 1.64775 | 1.03241 | **0.08914** | reject; step RMSE only, trajectory metrics worse |

## Full-data scale check

The following run uses all 213 test records, the same full-record protocol, `train_stride=10`, and initializes from the original full-data MEIO checkpoint. It is a fast screening run (`limit_batches=250` per epoch), so it is not the final training budget.

| variant | ATE XYZ (m) | RTE30 XYZ (m) | step RMSE (m) | decision |
|---|---:|---:|---:|---|
| full-data frozen MEIO reference | 1.85499 | 1.05890 | 0.09745 | reference |
| full-data three-domain residual, fast fine-tune, final exit | **1.69237** | **0.98231** | **0.09662** | keep as current full-data candidate |
| full-data three-domain residual, fast fine-tune, dynamic exit | 1.77941 | 1.03158 | 0.09876 | report separately; early exit trade-off |

Relative to the full-data MEIO reference, the final-exit candidate improves ATE by 8.8%, RTE30 by 7.2%, and one-step RMSE by 0.9%. This does not support a 30% claim yet. The run is recorded as a scale check rather than a final result because it used a capped batch budget.

The SSM replacement is implemented as a causal diagonal state recurrence. The stable candidate preserves the trained MLP and starts its SSM gate at zero, so initialization is function-preserving. It still did not improve the trajectory metrics in this screen. The direct replacement result is retained as a negative control because it demonstrates why a gated residual is needed.

Checkpoint and JSON reports are stored under `runs/` and `results/` locally; model weights are ignored by Git. The corresponding source configurations remain tracked so every row can be rerun.
