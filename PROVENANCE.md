# 来源、修改与许可证

本项目是官方公开代码基础上的 YT 数据适配和缺失组件重建。它不是作者发布的完整 MEIO-Net 实现，也不宣称复现论文表 III 的数字。

## MEIO-Net 官方代码

- 仓库：https://github.com/SunXuehang/MEIO-Net
- 本地 Git 历史起点：e93e5b82ef9cdb9920e13ab0c056f6525c5d4d2e，提交说明 fix dataset。
- 原始文件原样移到 upstream/EarlyExitPDRModule.py、upstream/PDRModule.py、upstream/README.md，保留历史；根 LICENSE 为官方 GPLv3。
- 官方公开的是 Lightning 训练模块和示例，依赖的 SmartPDR 数据加载、EarlyExitDense 主干、轨迹评估等未随该提交公开。因此不能直接调用一个完整的官方训练入口。

meio/train.py 和 meio/model.py 继承官方模块的多出口训练、MSE 向高斯 NLL 过渡、按深度加权、均值与 log 标准差双分支这些设计，改为独立 PyTorch 实现。修正了官方文件中只对 log sigma 项乘权重的写法：本地版本对整个 NLL 乘权重，与论文公式一致。出口权重为归一化的 (l/L)^0.5；不是兼容旧代码的错误加权协议。

## 官方致谢的 LLIO

- 仓库：https://github.com/i2Nav-WHU/LightweightLearnedInertialOdometer
- 获取时的树提交：e225ab8f3080c35c92481673b6af1d68692941f6。
- upstream/llio/model_MLP.py 和 LICENSE 原样保存，许可证 GPLv3。
- model_MLP.py Git blob：90bba2cb46406bafdd8a022d8af83d5dabf51594。

本地 Affine 和 Residual 改写自该文件的 Affine / PreAffinePostLayerScale，默认 8 层时使用 0.1 layer scale。patch 嵌入、token/channel 混合、多出口 head、逐样本动态路由和后续工具由本项目补齐。这里的 ResMLP 是有来源依据的重建，不是作者未公开的 EarlyExitDense 源码。原始 LLIO 文件作为溯源材料保存，不参与运行，也不要求安装它的全部依赖。

## 数据与论文

- 论文：MEIO-Net: A Motion-Aware Early-Exit Inertial Odometry Network for Efficient Pedestrian Dead Reckoning (PDR)，DOI 10.1109/JIOT.2025.3633690。
- 官方 README 指向 PedLocData v2：https://zenodo.org/records/18149105 。
- 文件 YT_server_200Hz.h5，15,138,253,808 字节；已核对本地 MD5 为 0f4559f4ef0542a99dc2bf5a45b958fe，与官方记录一致。
- 公开 H5 为 1479/422/213 条 train/valid/test 记录，合计 95.16906 h，论文写 108.69 h；本项目使用公开文件，不补造缺失记录。

源数据不随代码分发，数据许可由 Zenodo 记录约束。根目录衍生代码按 GPLv3 分发，保留两份上游源码和许可证。Git 的 upstream 指向作者官方仓库用于追溯，origin 指向用户指定的 https://github.com/pengkai123-whu/imu_work ，未向作者仓库推送修改。

## 明确采用的适配协议

1. 200 Hz，200 点输入，预测起点至 200 点后的三维位移；导航系输入，世界系标签；四元数 wxyz，设备到世界，保留重力。
2. 忽略发布 H5 的 para；仅由 train 重算导航系 mean/std，保存在缓存和 checkpoint。
3. 8 层、128 维、4 个 patch，三轴对角高斯不确定度。逐样本动态早退会裁剪已退出样本，未达阈值则走最终层。
4. 默认 20 轮 MSE、5 轮渐变、25 轮纯 NLL；固定 Adam 学习率 2.5e-4。best 仅从已经达到纯 NLL 的轮次中，按最终出口验证位移 RMSE 选择。
5. 增强包含偏航旋转、加速度/角速度偏置与噪声；没有实现官方示例的重力倾斜扰动，也不假装这些设置完全一致。
6. 验证集的两个不相交窗口子集分别校准 sigma 和选择阈值。默认允许相对最终出口最多 3% 的验证位移 RMSE 增加，在满足条件的阈值中选矩阵/卷积算量最低者；没有可用收益时回退最终出口。这与论文在训练集搜索阈值的协议不同。
7. 轨迹用不重叠 1 秒位移累加，输出 1 Hz 端点，报告 XY/XYZ ATE 和 30 秒 RTE，只有初始平移对齐。不能直接与论文未公开细节的 20 Hz 轨迹结果比较。
   当前测试主表采用所有完整、不重叠的 30 秒片段 ATE、完整记录内全合法起止对的 30 秒 RTE、单次 1 秒位移 RMSE；ATE 每段重置平移起点，尾段计数披露，完整记录指标保留。ATE 先按记录内平均再跨记录等权平均，未按误差选择片段；该分段协议不属于已确认的论文原评估设置。
8. 算量只统计主要 Linear/Conv 乘加（一次乘加计 2 FLOPs），包含动态推理访问过的所有 head。没有把 FLOPs 降幅换算成实际延迟收益。
