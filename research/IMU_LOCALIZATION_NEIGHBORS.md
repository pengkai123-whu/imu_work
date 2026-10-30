# Recent IMU/PDR neighbors used for module selection

This note records the 2024--2026 inertial-localization papers that constrain the next experiments. It complements `experiments/causal_tsf_odom/EXPERIMENT_LOG.md`; numbers from other datasets are not claimed as YT results.

| Work | Evidence-backed mechanism | What we can test on YT | Boundary |
|---|---|---|---|
| **FDIO**, arXiv 2511.15645 (2025), DOI `10.48550/arXiv.2511.15645` | Laplacian-pyramid low/high frequency split; Mamba for long-range low-frequency motion and multi-scale convolution for high-frequency detail | A causal low/high residual with a fixed state and a same-budget time-domain control | Existing multiscale and sample-SSM screens were negative; do not assume the Mamba name adds value |
| **EqNIO**, arXiv 2408.06321 (2024), DOI `10.48550/arXiv.2408.06321` | Gravity-aligned subequivariant processing and equivariant displacement/covariance | Body-frame plus attitude-aware branch, or a rigorously synchronized yaw/gravity equivariance loss | Current cache is already navigation-frame; a claim of EqNIO-style processing requires raw body-frame inputs and a coordinate test |
| **DIVE**, RA-L 2024, DOI `10.1109/LRA.2024.3370006` | IMU-history velocity regression used as a correction to an on-manifold EKF | A velocity auxiliary head and causal integration as an ablation before adding any EKF | Quadrotor dynamics and filter assumptions do not transfer automatically to pedestrian PDR |
| **AirIO**, RA-L 2025, DOI `10.1109/LRA.2025.3581130` | Body-frame observability and explicit attitude encoding, with learned IMU correction and uncertainty-aware filtering | Compare navigation-frame-only against body+attitude features under the same labels and split | UAV evidence is not pedestrian evidence; document the attitude source and online availability |
| **ResMixer**, IEEE Sensors Journal 2024, DOI `10.1109/JSEN.2024.3443311` | Lightweight residual channel/sequence mixing with fewer parameters than a deep CNN | Fair same-budget mixer baseline for the MEIO token MLP | Parameter count and FLOPs are not batch-1 latency; current patch mixer has no trajectory gain |
| **Revisiting Deep Inertial Odometry**, IEEE Sensors Journal 2025, DOI `10.1109/JSEN.2025.3634088` | EWT drift removal and a global-reference target are complementary drift experiments | Offline diagnostic and global-target ablation; keep main metrics causal and full-record | Test-trajectory EWT or reference fitting cannot enter the deployed method |

The broader survey **Deep Learning for Inertial Positioning**, IEEE T-ITS 2024, DOI `10.1109/TITS.2024.3381161`, is used only for the error-source taxonomy: calibration, drift reduction, and multi-sensor fusion. It does not justify a particular architecture.

The H5 inspection shows `acc`, `gyr`, `qua`, `pos`, `ts` plus `acc_bias`, `gyr_bias`, `mag`, and `motion_qua` for each record. The current cache intentionally keeps only navigation-frame six-axis signals, so the body-frame experiment must create a new cache identity and recompute train-only statistics; it cannot silently reuse old checkpoints. These fields are extra channels and attitude sources, not independent redundant IMUs.

The current decision is to keep the signed three-domain residual as the primary candidate, then test body-frame/attitude observability and a velocity auxiliary head one at a time. No redundant-IMU attention is added until the H5 file is shown to contain independently synchronized IMUs. No full-trajectory smoothing, EWT post-processing, or test-label adaptation is allowed in the main result.

The first body-frame/attitude screen is now implemented. It concatenates navigation-frame IMU, raw body-frame IMU, and quaternion (`nav6 + body6 + q4`) in a separate cache. A transfer-initialized 24-record screen reduced ATE from 1.61655 m to 1.57009 m, but RTE30 stayed at about 1.005 m and one-step RMSE increased from 0.08973 m to 0.09025 m. The module is therefore mixed and remains an ablation candidate, not a claimed contribution. The result also shows why raw extra channels must be evaluated with a function-preserving transfer initialization.

