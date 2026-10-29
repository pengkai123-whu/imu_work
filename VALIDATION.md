# 本地检查记录，2026-10-29

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

这些早期检查只证明执行路径，不证明网络收敛。后续已有 runs/yt 的 legacy 50 轮训练；新版 paper_fig3 和当前半秒更新配置尚未完成全量收敛训练。短训练 RMSE/ATE 不能作为论文结果。大 batch 合成吞吐不含加载和优化器开销；OOM 回退代码经过审查，没有人为耗尽显存实测。

## 当前 1 秒输入、1 秒位移、0.5 秒更新版本

12 项核心单测全部通过；新增图 3 共享 head、残差位置、patch 嵌入等价性、所有出口梯度、非零 log sigma 的完整 NLL 加权及重叠积分检查。恒速合成轨迹不会重复累加路程；加速轨迹真值用实际半秒端点差，能够检测窗口平均速度近似的误差。

validation/baseline_cadence_execution.json 记录真实 H5、CUDA、完整 8x128 paper_fig3 的两种更新配置。每个 split 2 条记录、每条 14000 点，4 轮每轮 3 个 batch，并恢复到第 5 轮；MSE、过渡、NLL、valid 校准、train 原始标准差选阈值、final/dynamic 评估均通过，AMP 未跳过更新。默认半秒更新测试产生 276 个 1 秒预测、274 个半秒轨迹区间和 4 个完整 30 秒片段。该结果不是精度实验。

旧 legacy 权重另在全部 213 条记录上复测，位移 RMSE、完整 ATE/RTE、出口分布与三个主指标保持原值。旧模型诊断见 validation/baseline_diagnostics.json，不能混作新结构证据；一次 batch=1 的 GPU 诊断中动态路由慢于最终出口，不宣称实际时延收益。

test.py 新默认配置的 CLI、final_only 仅最终出口 CLI、train 原始标准差参考 checkpoint 阈值选择 CLI 也通过；汇总见 validation/cadence_cli_checks.json。协议说明文字有更新，旧模型兼容检查针对数值字段。

本地原始日志、配置、checkpoint 和 JSON 报告保存在 runs/，不进入 Git。全量缓存保存在 cache/yt，可供本地后续分析。当前启动方式为 python train.py / python test.py。

后续正式训练模型的固定 60 秒补充测试使用已冻结的 runs/yt/policy.json，未重新训练、校准或按测试误差挑选窗口。515 个完整片段覆盖全部 213 条记录，覆盖 30,900/36,105 个预测区间，5,205 个尾段区间只从补充指标排除。动态早退的完整 XYZ ATE / RTE30 为 1.936220 / 1.103933 m；60 秒片段为 1.108257 / 1.058129 m。这是评估范围变化，不是模型精度提升，也不能直接与论文数值比较。机器可读结果见 validation/fixed_duration_evaluation.json，完整分段明细位于 runs/yt/evaluation_60s/。

历史主表曾按用户约定采用 30 秒片段 ATE、完整记录内全起点 30 秒 RTE、单次 1 秒位移 RMSE。新增第 8 项已知偏置测试，8 项全部通过；相同模型与冻结策略在 CUDA 上复测全部测试记录。30 秒 ATE 共 1108 段、213 条记录，覆盖 33240/36105 个区间（92.06%），余下 2865 个区间只从片段 ATE 排除。动态早退三个 XYZ 指标分别为 0.670088 / 1.103933 / 0.099665 m。完整记录指标、出口分布、位移 RMSE 均与之前逐值一致。见 validation/primary_metrics_30s.json 与 runs/yt/evaluation_30s/summary.json。

后续仅调整测试主指标：完整轨迹 ATE、RTE30、Disp1s RMSE，protocol_version=2。不修改训练或积分器，分段 ATE 留作补充。旧正式模型在 CUDA 上重测全部 213 条记录，新版 0.5 秒更新的小切片 checkpoint 同样完成 CLI 检查；结果与数值一致性记录见 validation/full_ate_primary.json。新版小切片的数值不作收敛精度。
