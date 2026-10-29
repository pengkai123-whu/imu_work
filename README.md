# MEIO-YT：本地 GPU 训练与测试

基于 MEIO-Net 公开训练模块和 LLIO 参考代码补齐的 YT 适配实现，全部入口为 Python。官方未发布完整主干及数据/轨迹代码，因此本项目属于框架重建与适配，不能宣称严格复现论文数字。来源见 [PROVENANCE.md](PROVENANCE.md)，核心核对和局限见 [BASELINE_AUDIT.md](BASELINE_AUDIT.md)。

## 当前默认版本

配置 configs/yt_1s_update05s.json：200 Hz，输入过去 1 秒六轴 IMU（200 点），预测同一秒的世界系三维位移及不确定度，每隔 0.5 秒更新（100 点步长，2 Hz）。例如 t=1.0 输入 [0,1.0)，预测 p(1.0)-p(0)；t=1.5 输入 [0.5,1.5)，预测 p(1.5)-p(0.5)。并非预测未来一秒，也并非半秒输入。

4 个 patch、8 层、128 维，每层有位移和不确定度出口；新版 paper_fig3 使用图示的共享隐藏层 head，并调整残差支路 Affine 的位置。旧 legacy 结构和 runs/yt 权重保持兼容，但不能用于新版结构续训或充当新版成绩。

## 训练与测试

~~~powershell
conda activate spk
cd C:\Users\19510\Desktop\work_space\imu_work\meio_yt
python train.py
python test.py
~~~

默认训练读取上一级 YT_server_200Hz.h5，复用 cache/yt，在本地 CUDA 上训练，输出 runs/yt_1s_update05s。无需上传数据或安装 Jupyter/Lightning。完整配置没有记录、采样或 batch 数量裁剪。训练每 10 点采一个窗口以增加训练样本，验证/阈值选择/测试每 100 点采一个窗口。窗口密度和在线更新间隔是不同设置。

训练为 20 轮 MSE、5 轮过渡、25 轮 NLL；gamma=0.5 的深度权重作用于完整损失。best.pt 仅从纯 NLL 阶段按最终出口验证 RMSE 选择，last.pt 用于恢复，mse_last.pt 保存均值预训练。metrics.jsonl 记录每轮损失、各出口验证误差和耗时。

~~~powershell
python train.py --resume
python train.py --batch 2048 --no-auto-batch
python train.py --out runs/new_experiment
python test.py --checkpoint runs/new_experiment/best.pt --out runs/new_experiment/reports
~~~

自动 batch 默认从 2048 请求值开始探测，最大 8192，按合成前反向吞吐和 80% 显存条件选择；真实训练 OOM 会恢复本轮初始状态、减半 batch 重试。探测不含加载和优化器开销，大 batch 不保证收敛或实际吞吐更好。已有 last.pt 不带 --resume 会拒绝覆盖。相对配置路径按 JSON 目录解析，CLI 路径按终端当前目录解析；更换数据内容要指定新的缓存目录。

## 重叠预测、轨迹与指标

相邻预测覆盖同一秒中的部分时间，不能整段相加。采用窗口平均速度近似：每次用 1 秒预测位移除以 1 秒，再乘 0.5 秒推进轨迹。第一条输出在 t=1 秒初始化轨迹原点；之后从 t=1.5 秒开始推进。真实轨迹增量使用实际 p(t)-p(t-0.5s)。加速/转弯时，平均速度近似本身存在误差；报告 integration_oracle 用精确 1 秒真值输出通过同一积分器计算诊断值，不当作网络精度或误差下界。

终端主表固定三个 XYZ 指标（米）：

- **完整轨迹 ATE**：在每条记录的全部可重建轨迹端点上计算位置 RMSE，包含尾部；只对齐初始位置，中途不重置，不拟合旋转或尺度。先每条记录内计算 RMSE，再跨记录等权平均。
- **RTE30s**：重建轨迹上全部合法 30 秒端点对，起点每 0.5 秒一个；每条记录内 RMSE 后跨记录等权平均，包含尾部合法起止对。
- **Disp1s RMSE**：每隔 0.5 秒输出的完整 1 秒位移向量误差，汇总全部窗口平方误差后开方。不是半秒位移误差。

30 秒分段 ATE 留作补充，位于 JSON 的 fixed_duration，并在终端单独展示。--segment-seconds 只改变补充分段，不改变主表完整 ATE。主表 JSON 的 primary_metrics 使用 protocol_version=2、ate_scope=full_record；旧报告的同名 ate_xyz 可能是分段值，汇总时必须核对协议。分段指标每段重置平移起点，排除不足时长尾段并披露覆盖率；主表完整 ATE 不排除该尾段。

XY 指标、逐记录数据、各出口分布和 Linear/Conv FLOPs 都保留在 JSON。RTE 报告可用和不足 30 秒的记录数。FLOPs 同时报告每次输出和每秒值，不把减少 FLOPs 直接解释为 GPU 加速。当前更新与积分协议不能直接与论文的 20 Hz 轨迹结果比较；所有对比方法应采用相同协议。

## 早退策略与消融

python test.py 默认先在 valid 的两半窗口上分别校准标准差、搜索阈值，然后冻结策略，分别评估 final 和 early。允许位移 RMSE 相对最终出口增加最多 3%，满足条件的阈值中选择主要矩阵算量最低者，无合适收益则回退 final。max-windows 默认 50000，只限制阈值选择，不裁剪测试。

~~~powershell
python test.py --policy runs/yt_1s_update05s/policy.json --out runs/yt_1s_update05s/retest
python tools/make_ablations.py
python train.py --config configs/yt_1s_update05s_final_only.json
python train.py --config configs/yt_1s_update05s_equal_weights.json
~~~

final_only 只监督最终出口，测试只跑 final，不使用未训练中间 head；equal_weights 使用 gamma=0。它们与默认模型拥有相同数据和优化设置，各自独立保存。final_only 保留未使用中间 head 参数，论文统计参数量时需说明。正式报告建议至少三个随机种子，不能用执行检查替代收敛消融。

接近论文 III-C 的训练集原始不确定度搜索与参考精度匹配，可在训练好默认和 final_only 后运行：

~~~powershell
python test.py --selection-split train --raw-sigma --tolerance 0 --reference-checkpoint runs/yt_1s_update05s_final_only/best.pt --out runs/yt_1s_update05s/train_raw_policy
~~~

参考模型在完全相同的选阈值窗口上测量；默认最多均匀抽取 50000 个窗口。标量不确定度采用各轴标准差 L2 范数，这也是未公开细节下的本地选择，不能声称数值阈值与作者完全相同。测试集不参与校准、阈值或参考精度选择。

## 旧版本与本地检查

configs/yt_paper_1s.json 保留 1 秒非重叠参考配置（窗口/主干参数参考论文，轨迹 1Hz 并非完整论文协议）。configs/yt.json 与 runs/yt/best.pt 是历史 legacy 实验，需显式指定：

~~~powershell
python test.py --checkpoint runs/yt/best.pt --policy runs/yt/policy.json --out runs/yt/new_report
python -m unittest discover -s tests -v
python tools/check_baseline.py
~~~

check_baseline 使用真实 H5 小切片在 CUDA 上运行训练、恢复、两种选阈值方式及重叠评估；已有输出会拒绝覆盖，重测先改脚本 out。详情见 [VALIDATION.md](VALIDATION.md)。新默认版本已通过执行检查，尚未完成全量收敛训练。数据、缓存、权重、运行日志不进 Git，原始 H5 不修改。

公开数据 1479/422/213 条 train/valid/test 记录，约 95.17h，论文写 108.69h；不是受试者互斥划分。设备系 IMU 通过发布 wxyz 姿态旋转到世界系，保留重力；仅 train 计算导航系归一化，忽略 H5 的 para。姿态是额外依赖，不应声称纯六轴无外部姿态输入。

代码保留官方历史和 GPLv3 许可；origin 为用户仓库，upstream 为作者仓库，不向 upstream 推送。
