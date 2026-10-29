# 本地检查记录，2026-10-29

当前恢复 5192f95 的旧结构、训练入口和配置，使用已完成训练的 runs/yt/best.pt，不重新训练。测试保留 protocol_version=2 的完整 ATE、RTE30、Disp1s RMSE 主指标，分段 ATE 为补充。默认复用 runs/yt/policy.json。恢复后的 CUDA 全量复测记录见 validation/restored_1s.json；历史阶段记录如下，不应将其中旧的分段主表当作当前主表。0.5 秒版代码在 Git 历史，未删除其本地实验文件。

检查在现有 spk 环境的真实 CUDA GPU 上运行：D:/anaconda/envs/spk/python.exe，PyTorch 2.9.0.dev20250813+cu129，NVIDIA GeForce RTX 5060 Laptop GPU，约 7.96 GiB 显存。机器可读记录见 [validation/evidence.json](validation/evidence.json)。

| 检查 | 实际结果与范围 |
|---|---|
| Python 编译与 CLI | compileall 通过；run.py --help 正常 |
| 本地 Python 入口 | train.py 在 CUDA 上完成真实 H5 小切片训练；train.py --resume 自动恢复到第 5 轮；test.py 完成验证校准、最终出口和动态测试、生成 summary.json；从项目外目录调用也通过 |
| 6 项核心测试 | 全通过；覆盖四元数旋转、输入/标签边界、仅训练集统计、混合早退裁剪、完整 NLL 加权、轨迹累加与 30 秒 RTE、增强形状 |
| 固定时长补充评估 | 新增第 7 项测试，验证全片段纳入、尾段计数、重置平移、已知偏置下 ATE/RTE 数值和短记录；7 项全部通过。现有正式模型在 CUDA 上复测全部 213 条记录、36,105 个窗口，原完整记录指标及出口分布逐值不变 |
| 真实 H5 小切片 GPU smoke | train/valid/test 各 2 条记录、每条前 8000 点；完整 8 层 128 维模型；4 轮、每轮 3 个 batch；保存和恢复到第 4 轮、校准、最终出口与动态推理均通过，AMP 跳过更新为 0 |
| 大 batch | 128/512/2048/4096/8192 探测完成，本次选择 8192；合成探测峰值 reserved 约 1.36 GiB；真实数据 batch=8192 的 MSE/混合/NLL 更新均通过，检查了参数确实改变 |
| 全量缓存 | 全部 2114 条记录、68,521,724 点通过有限值、时间间隔、四元数及长度检查；train/valid/test 为 1479/422/213；train-only mean/std 与此前独立审计吻合 |
| 全量缓存短训练 | 4,710,576 个可用训练窗口、69,218 个验证窗口；实际仅每轮取 3 个 batch，batch=8192；NLL 比例 0、0.5、1、1 均通过；没有 AMP 跳过更新 |
| 全量校准与评估执行 | 用短训练 checkpoint 对 50,000 个验证窗口完成校准/选阈值；全部 213 条测试记录、36,105 个非重叠窗口的最终出口及动态评估均完成 |
| 上游完整性 | MEIO 原始三文件与基提交逐字节比较一致；LLIO model_MLP.py Git blob 为 90bba2cb46406bafdd8a022d8af83d5dabf51594 |

运行入口：

~~~powershell
python -m unittest discover -s tests -v
python tools/check_local.py
python tools/smoke_check.py --device cuda --out runs/new_smoke
python tools/gpu_batch_check.py --out runs/new_batch_check
python tools/full_data_check.py
~~~

最后一个脚本固定写入 runs/full_data_check，已有 last.pt 时会拒绝覆盖；重测应先改输出目录。完整实验使用 configs/yt.json，不能把检查脚本的 limit_batches 带入正式结果。

上述早期执行检查不证明网络收敛。后续 runs/yt 已有完整 50 轮训练权重，本次直接复用，不重新训练；短训练检查的 RMSE/ATE 仍不能当作正式精度。大 batch 的合成吞吐不包含加载和优化器开销，不作为训练速度结论。CUDA OOM 回退代码已审查，但没有人为制造显存耗尽来实测该分支。

本地原始日志、配置、checkpoint 和 JSON 报告保存在 runs/，不进入 Git。全量缓存保存在 cache/yt，可供本地后续分析。当前启动方式为 python train.py / python test.py。

后续正式训练模型的固定 60 秒补充测试使用已冻结的 runs/yt/policy.json，未重新训练、校准或按测试误差挑选窗口。515 个完整片段覆盖全部 213 条记录，覆盖 30,900/36,105 个预测区间，5,205 个尾段区间只从补充指标排除。动态早退的完整 XYZ ATE / RTE30 为 1.936220 / 1.103933 m；60 秒片段为 1.108257 / 1.058129 m。这是评估范围变化，不是模型精度提升，也不能直接与论文数值比较。机器可读结果见 validation/fixed_duration_evaluation.json，完整分段明细位于 runs/yt/evaluation_60s/。

当前主表按用户约定改为 30 秒片段 ATE、完整记录内全起点 30 秒 RTE、单次 1 秒位移 RMSE。新增第 8 项已知偏置测试，8 项全部通过；相同模型与冻结策略在 CUDA 上复测全部测试记录。30 秒 ATE 共 1108 段、213 条记录，覆盖 33240/36105 个区间（92.06%），余下 2865 个区间只从片段 ATE 排除。动态早退三个 XYZ 指标分别为 0.670088 / 1.103933 / 0.099665 m。完整记录指标、出口分布、位移 RMSE 均与之前逐值一致。见 validation/primary_metrics_30s.json 与 runs/yt/evaluation_30s/summary.json。
