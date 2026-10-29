# MEIO-YT：Colab 训练与本地 GPU 检查

这是针对现有 YT_server_200Hz.h5 的 MEIO 风格多出口惯性里程计版本。代码基于 MEIO-Net 官方仓库历史和其致谢的 LLIO 残差模块补齐，主入口是 MEIO_YT_Colab.ipynb。完整来源和与论文的差异见 [PROVENANCE.md](PROVENANCE.md)。

你手里的 H5 已核对为作者正式公开文件；公开数据是 95.17 小时，与论文 108.69 小时存在版本差异。当前任务采用这份公开数据建立可运行的适配基线，训练结果需要你完成全量训练后再判断。

## Colab 上怎么运行

1. 将交付的 meio_yt_colab.zip 上传到 Google Drive 的 MyDrive/MEIO/。另把 MEIO_YT_Colab.ipynb 在 Colab 打开。
2. 选择“运行时 → 更改运行时类型 → GPU”，按顺序执行笔记本。第一段会打印实际 GPU、CUDA 和显存；参考了原 colab_gpu_test.ipynb 的环境检查方式。
3. 路径单元格默认 DATA_SOURCE="drive"：把 H5 也放到 MyDrive/MEIO/。如果不想上传约 15 GB 的文件，改成 "official"，直接从官方 Zenodo 下载到云端。需要约 25 GiB 空闲磁盘，H5 不进入 Git。
4. 保留 RUN_NAME="yt_full_v1" 可接续该实验；新实验换名称。首次先运行 GPU smoke 检查，再生成全量缓存、训练、校准和评估。
5. 运行参数在标注“16GB Colab”的单元格里：batch=2048，auto_batch=True，max_batch=8192。程序也会探测 128、512 等较小候选，选显存占用低于总量 80% 且实测吞吐较高的候选。发生 CUDA OOM 时会恢复本轮开始的状态并减半 batch 重试。

batch 探测包含 GPU 前向、反向和预热，不包含数据加载吞吐，也不保证 Colab 与本机选择相同。大 batch 不会自动放大学习率。实际训练速度看 metrics.jsonl 的每轮耗时；更大的 batch 可能影响优化和泛化，第一轮正式实验建议先保留默认学习率。

笔记本缓存放 /content 的本地磁盘；checkpoint、配置、指标、校准策略和评估 JSON 每 30 秒及每条命令结束同步到 Drive。保存发生在完整 epoch 结束，硬断线后最多损失尚未完成的一轮及尚未同步的最近 checkpoint。重新连接后依次运行环境、代码、数据和缓存单元格，再执行训练单元格，会从 last.pt 续训。校准依赖 best.pt，务必保留整个运行目录。

Drive 挂载需在 Colab 里由你授权。本地检查不能代替在你的 Colab runtime 中执行；本项目没有替你登录云端账号。

## 本地 spk 环境

在 meio_yt 目录打开终端，使用现有 spk 环境；无需安装 Lightning。

~~~powershell
conda activate spk
python -m unittest discover -s tests -v
python tools/smoke_check.py --device cuda --out runs/my_gpu_smoke
~~~

也可用 spk kernel 打开同一份笔记本。第一格会使用本地 GPU；完整训练建议留给 Colab。本地运行路径默认找到上一级的 H5。若缺 numpy / h5py，可安装 requirements.txt；已有支持本机显卡的 torch 时，避免随意替换其 CUDA 版本。

命令行入口与笔记本共用代码，配置中的相对路径相对于 JSON 所在目录解析：

~~~powershell
python run.py prepare --config configs/yt.json
python run.py train --config configs/yt.json
python run.py train --config configs/yt.json --resume runs/yt/last.pt
python run.py calibrate --checkpoint runs/yt/best.pt --cache cache/yt --out runs/yt/policy.json
python run.py evaluate --checkpoint runs/yt/best.pt --cache cache/yt --out runs/yt/final_test.json
python run.py evaluate --checkpoint runs/yt/best.pt --cache cache/yt --policy runs/yt/policy.json --out runs/yt/early_test.json
~~~

上面 train 与 resume 是两种启动方式；已有 last.pt 时不带 resume 会拒绝覆盖。默认完整训练为 50 轮，默认配置没有数据裁剪和 limit_batches。smoke 脚本会单独裁剪数据并限制批次数，不影响正式配置。

## 数据与结果含义

- 仅使用 H5 的 train/valid/test 原始划分，未改成受试者互斥划分；不能据此宣称未见受试者泛化。
- 缓存检查有限值、四元数和 200 Hz 时间间隔，旋转到世界系后仅从 train 估计 mean/std。原 H5 不修改。
- 输入 1 秒 IMU，输出对应 1 秒 XYZ 位移。窗口同时检查输入与标签边界，避免短窗口混入 batch。
- 先学习多出口均值，再训练对角高斯 NLL；权重乘完整 NLL。校准用验证集，测试集不参与阈值选择。
- early 推理对当前活跃样本逐层计算、满足条件就退出，记录各出口数量；如果没有符合精度约束且省算量的阈值，策略自动用最终出口。
- 测试报告含位移 RMSE、每条轨迹及宏平均的 XY/XYZ ATE、30 秒 RTE、出口分布和矩阵/卷积 FLOPs。30 秒不足的记录其 RTE 为 null。
- 轨迹按不重叠 1 秒位移累加，原点平移对齐，无额外旋转拟合。这是明确的 1 Hz 评估协议，不能用来声称复现论文 20 Hz 数值。

输出文件：last.pt 用于续训，best.pt 用于校准和评估，mse_last.pt 保留均值预训练结束前的模型；metrics.jsonl 记录每轮损失、各出口验证 RMSE、batch、AMP 跳过步数和耗时；batch_probe.json 保存显存与吞吐探测；policy.json 绑定 checkpoint SHA256；final_test.json 和 early_test.json 用于比较最终出口与动态路由。

## 本地与云端协同

推荐本地做检查、分析，Colab 做正式训练；或两端用不同 RUN_NAME 同时做不同实验。需要交接训练时复制整个运行目录，并使用同一数据和缓存版本；checkpoint 保存优化器、AMP scaler、随机状态、模型配置和归一化参数。路径和 batch 可变，关键设置不匹配会拒绝恢复。跨环境重建缓存若身份检查不一致，应复制完整缓存及 manifest，不能手工改 identity 绕过检查。smoke 子集 checkpoint 不能用于全量实验续训。

这套协作不通过公网同步两张卡的梯度。两端不能同时写同一个实验目录。训练恢复在 epoch 边界，跨硬件和不同 batch 不承诺逐比特一致。

## 代码和验证

meio/ 为实际运行代码，upstream/ 保留官方原件，tests/ 为关键语义检查，tools/ 包含 notebook 生成、真实数据 GPU smoke、大 batch 检查和打包脚本。检查范围和已运行记录见 [VALIDATION.md](VALIDATION.md)。

独立仓库分支 yt-adaptation，保留官方 Git 历史；origin 只用于来源追溯，没有向官方仓库推送。代码包不包含数据、缓存、权重或 .git；需要历史时使用本地仓库。代码为 GPLv3，数据许可见官方 Zenodo。
