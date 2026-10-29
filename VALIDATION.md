# 本地检查记录，2026-10-29

检查在现有 spk 环境的真实 CUDA GPU 上运行：D:/anaconda/envs/spk/python.exe，PyTorch 2.9.0.dev20250813+cu129，NVIDIA GeForce RTX 5060 Laptop GPU，约 7.96 GiB 显存。机器可读记录见 [validation/evidence.json](validation/evidence.json)。

| 检查 | 实际结果与范围 |
|---|---|
| Python 编译与 CLI | compileall 通过；run.py --help 正常 |
| 本地 Python 入口 | train.py 在 CUDA 上完成真实 H5 小切片训练；train.py --resume 自动恢复到第 5 轮；test.py 完成验证校准、最终出口和动态测试、生成 summary.json；从项目外目录调用也通过 |
| 6 项核心测试 | 全通过；覆盖四元数旋转、输入/标签边界、仅训练集统计、混合早退裁剪、完整 NLL 加权、轨迹累加与 30 秒 RTE、增强形状 |
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

这些结果证明当前环境下关键路径已经实际执行，不证明网络已收敛。没有完成默认 50 轮全量训练，短训练得到的 RMSE/ATE 不能当作论文复现结果。大 batch 的合成吞吐不包含加载和优化器开销，不作为训练速度结论。CUDA OOM 回退代码已审查，但没有人为制造显存耗尽来实测该分支。

本地原始日志、配置、checkpoint 和 JSON 报告保存在 runs/，不进入 Git。全量缓存保存在 cache/yt，可供本地后续分析。当前启动方式为 python train.py / python test.py。
