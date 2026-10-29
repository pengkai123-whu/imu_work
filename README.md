# MEIO-YT：本地 GPU 训练与测试

针对本地 YT_server_200Hz.h5 的多出口惯性里程计适配版本。训练和测试入口均为 Python 脚本，直接在终端或 VS Code 运行。代码基于 MEIO-Net 官方公开训练模块及其致谢的 LLIO 残差模块补齐，来源与论文差异见 [PROVENANCE.md](PROVENANCE.md)。

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

它读取 runs/yt/best.pt，先在验证集校准不确定度并选择早退阈值，然后分别在全部测试记录上评估最终出口和动态早退。测试集不参与选阈值。结果保存到 checkpoint 同目录的 policy.json、final_test.json、early_test.json、summary.json。

主输出为三个指标（米）：30 秒片段 ATE、30 秒 RTE、单次 1 秒位移 RMSE。终端主表使用 XYZ 向量误差，JSON 的 primary_metrics 同时保存 XY 结果与协议。ATE 从每条记录零时刻起连续切分，采用全部完整、不重叠的 30 秒片段，每段只重置起点平移，不做旋转或尺度拟合；不足 30 秒的尾段不进入片段 ATE，报告披露覆盖率。先在每条记录内平均片段 ATE，再跨记录等权平均。30 秒 RTE 采用完整记录内所有合法起止对（每秒一个起点），每条记录内计算 RMSE 后跨记录等权平均，不受 ATE 分段影响，也包含尾段中的合法起止对。单次位移 RMSE 在全部 1 秒预测上汇总平方误差后开方。完整记录 ATE 与片段内 RTE 保留在详细 JSON，分别位于 macro / fixed_duration 中，不能与主输出混用。

自定义实验：

~~~powershell
python test.py --checkpoint runs/experiment2/best.pt
python test.py --checkpoint runs/experiment2/best.pt --batch 256 --out runs/experiment2/reports
python test.py --checkpoint runs/yt/best.pt --policy runs/yt/policy.json --out runs/yt/evaluation_30s
~~~

--max-windows 只限制验证集校准窗口数，不裁剪测试集。默认最多 50000 个验证窗口，允许相对最终出口最多 3% 的验证位移 RMSE 增加；没有符合精度与算量条件的阈值时自动用最终出口。

--policy 可复用与 checkpoint 绑定的已冻结早退策略，避免重新校准；--segment-seconds 默认 30 秒，只改变 ATE 片段时长，RTE 间隔始终是 30 秒。片段时长至少 30 秒且为预测间隔的整数倍，应在比较模型前统一确定，不根据测试误差挑选最有利的时长、起点或片段。重新评估建议指定独立 --out，以保留旧报告。30 秒片段 ATE 是当前任务约定，不是已确认的论文原协议，也不表示连续运行的完整轨迹误差。

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
