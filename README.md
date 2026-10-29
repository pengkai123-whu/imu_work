# MEIO-YT：本地 GPU 训练与测试

针对本地 YT_server_200Hz.h5 的多出口惯性里程计适配版本。训练和测试入口均为 Python 脚本，直接在终端或 VS Code 运行。代码基于 MEIO-Net 官方公开训练模块及其致谢的 LLIO 残差模块补齐，来源与论文差异见 [PROVENANCE.md](PROVENANCE.md)。

当前按用户选择恢复到历史提交 5192f95 的模型、训练和 1 秒非重叠更新配置，额外保留完整轨迹 ATE 主指标。已有 runs/yt/best.pt 与冻结 policy.json 可直接测试，不需要重新训练。0.5 秒版代码仍保存在历史提交 6f7f133，实验权重和日志仍在 runs/yt_1s_update05s；本次不删除这些本地实验文件。旧结构与图 3 重建结构存在差异，见来源说明，不能将旧权重称为严格论文复现。

## 开始训练

在终端执行：

~~~powershell
conda activate spk
cd C:\Users\19510\Desktop\work_space\imu_work\meio_yt
python train.py
~~~

默认使用本机 CUDA GPU，从上一级目录读取 YT_server_200Hz.h5，自动准备或复用 cache/yt，然后开始训练。已经生成的完整缓存可直接复用。无需 Jupyter 或 Lightning，也不需要上传数据。当前 spk 已有可运行的 CUDA PyTorch；不要随意替换其版本。缺少依赖时参考 requirements.txt。

参数统一在 configs/yt.json：默认 200 Hz、200 点输入、1 秒 XYZ 位移、8 层 128 维、20 轮 MSE + 5 轮过渡 + 25 轮 NLL。完整训练没有数据裁剪或批次限制，输出目录为 runs/yt。每 20 个 batch 打印进度，每轮打印验证结果。

batch 从 2048 起自动探测，最大 8192，按合成前向/反向吞吐及低于 80% 显存条件选取；真实训练遇到 CUDA OOM 时恢复本轮开始状态、减半 batch 重试。探测吞吐不含加载和优化器开销，大 batch 可能影响优化效果，学习率保持 2.5e-4。固定 batch 可运行：

~~~powershell
python train.py --batch 2048 --no-auto-batch
~~~

中断后从最近一个完整 epoch 恢复：

~~~powershell
python train.py --resume
~~~

另开实验或指定数据：

~~~powershell
python train.py --out runs/experiment2
python train.py --h5 "C:\path\YT_server_200Hz.h5"
~~~

更换数据内容时同时在配置中指定新的 cache 目录。配置中的相对路径相对于 JSON 文件所在目录解析，命令行覆盖的路径相对于当前终端目录解析。已有 last.pt 时不带 --resume 会拒绝覆盖。

## 测试与结果

训练完成后执行：

~~~powershell
python test.py
~~~

它读取 runs/yt/best.pt，默认复用同目录已有的冻结 policy.json，分别在全部测试记录上评估最终出口和动态早退；只有找不到已有策略，或明确指定 --recalibrate 时才在验证集校准与选阈值。测试集不参与选阈值。默认报告保存在 runs/yt/evaluation_full，避免覆盖原始报告。

主表为完整轨迹 ATE、30 秒 RTE、单次 1 秒位移 RMSE（米）。200 Hz、200 点输入，预测同一秒位移，每隔 1 秒更新，非重叠位移直接累加。完整 ATE 包含全部可重建轨迹端点和尾部，只做一次起点平移对齐，中途不重置、不拟合旋转或尺度。ATE 与 RTE 先按记录计算 RMSE，再跨记录等权平均；RTE 使用每条完整轨迹中全部合法 30 秒起止对，每秒一个起点。位移 RMSE 汇总全部预测窗口平方误差。JSON 的 primary_metrics.protocol_version=2、ate_scope=full_record；XY 和原始逐记录数据仍保留。

30 秒片段 ATE 单独作为 fixed_duration 补充展示，全部完整片段从记录起点连续切分，每段仅重置平移，尾段排除和覆盖率披露；它不代替主表完整 ATE。历史报告的主表 ATE 可能是分段值，比较前必须核对协议。

自定义实验：

~~~powershell
python test.py --checkpoint runs/experiment2/best.pt
python test.py --checkpoint runs/experiment2/best.pt --batch 256 --out runs/experiment2/reports
python test.py --checkpoint runs/yt/best.pt --policy runs/yt/policy.json --out runs/yt/evaluation_30s
~~~

--max-windows 只限制验证集校准窗口数，不裁剪测试集。默认最多 50000 个验证窗口，允许相对最终出口最多 3% 的验证位移 RMSE 增加；没有符合精度与算量条件的阈值时自动用最终出口。

--policy 可显式指定 checkpoint 绑定的冻结策略；默认会复用权重同目录的策略。--recalibrate 才强制重新选择，并将新策略写入报告目录。--segment-seconds 默认 30 秒，只改变补充分段 ATE，不影响三个主指标。片段时长至少 30 秒且为预测间隔的整数倍，不按测试误差挑片段。

last.pt 用于续训，best.pt 仅从达到纯 NLL 阶段的模型中按最终出口验证 RMSE 选择，mse_last.pt 保留均值预训练模型。metrics.jsonl 记录损失、各出口验证 RMSE、batch、AMP 跳过步数和耗时，batch_probe.json 保存显存与吞吐探测。每轮结束保存 checkpoint，未完成一轮的进度不保存。

## 本地检查

~~~powershell
python -m unittest discover -s tests -v
python tools/smoke_check.py --device cuda --out runs/my_gpu_smoke
~~~

smoke 是真实 H5 小切片的训练、恢复、校准和评估检查，不能当作精度实验。输出目录需使用新名称。已有本地检查见 [VALIDATION.md](VALIDATION.md)。

run.py 保留分步命令 prepare / train / calibrate / evaluate，便于独立执行其中一项；用 python run.py --help 查看。

## 数据和协议

- H5 是作者官方公开版本，1479/422/213 条 train/valid/test 记录，共约 95.17 小时；论文写 108.69 小时，不能据公开文件保证复现论文数字。原始划分不是受试者互斥划分。
- wxyz 四元数从设备系旋转到世界系，IMU 保留重力，三维位移为世界系标签。仅使用 train 重算导航系 mean/std，忽略发布 H5 的 para，统计量和缓存身份绑定 checkpoint。原 H5 不修改。
- 多出口高斯 NLL 权重乘完整损失；逐样本动态路由会移除已退出样本，并计入所有访问 head 的算量。
- 轨迹使用不重叠 1 秒位移累加，1 Hz 端点，只有初始平移对齐。报告位移 RMSE、XY/XYZ ATE、30 秒 RTE、出口分布及主要 Linear/Conv FLOPs。短于 30 秒的记录 RTE 为 null。该协议不能直接与论文的 20 Hz 轨迹数字比较。

## Git 与源码

train.py / test.py 是本地入口；meio/ 是模型、数据和训练实现；upstream/ 保存原始来源与许可证；tests/ 和 tools/ 为检查工具。数据、缓存、运行日志、模型权重、Python 缓存不进入 Git。衍生代码 GPLv3，数据许可见官方 Zenodo。保留官方历史，在独立分支开发，不向作者仓库推送修改。
