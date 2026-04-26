## Plan: 耕地分割论文素材整理与改进路线

聚焦 farm-u-net 与 过程文件/U-NET 的深度学习分割主线。先输出“代码可确认”的模型结构与指标口径，再给出可执行的改进实验矩阵（结构、训练策略、矢量化后处理），将具体数值留待补测填表。

**Steps**
1. 固化当前模型技术画像（*先行步骤*）：基于现有源码整理主干网络、注意力模块、多任务头、损失函数、训练策略与推理后处理，形成论文“方法”小节素材。
2. 构建当前精度证据表（*depends on 1*）：区分“已在代码中明确定义的指标口径”和“尚未落盘的具体分数”。
3. 设计精度补测方案（*depends on 2*）：定义最小补测任务，仅补齐 best_iou、IoU/Dice/F1/Precision/Recall 五类核心指标，并统一阈值与数据划分口径。
4. 设计改进实验 Phase A（结构改进，*depends on 1，可与5并行*）：围绕输入通道扩展（多光谱）、注意力模块替换、边界分支强化做消融。
5. 设计改进实验 Phase B（训练策略改进，*depends on 1，可与4并行*）：围绕损失函数替换、困难样本挖掘、阈值校准与TTA做消融。
6. 设计改进实验 Phase C（后处理与矢量化，*depends on 1，可与4/5并行*）：围绕边界腐蚀、形态学核、轮廓简化 epsilon、最小面积过滤做网格实验，并记录矢量质量指标。
7. 汇总论文实验章节骨架（*depends on 4/5/6*）：形成“基线 + 消融 + 对比 + 误差分析”结构，明确每个表格与图的来源字段。

**Relevant files**
- d:/Work space/DeepLearning/farm/farm-u-net/u-net--CBAMV7.py — 主模型结构、训练循环、验证 IoU、checkpoint 保存字段。
- d:/Work space/DeepLearning/farm/farm-u-net/u-net矢量化.py — 读取 checkpoint 的 best_iou/epoch 并执行矢量化推理。
- d:/Work space/DeepLearning/farm/过程文件/U-NET/best_model_config.yaml — 历史最佳配置记录（encoder/lr/epochs/use_cbam 等）。
- d:/Work space/DeepLearning/farm/过程文件/U-NET/u-net分割推理--cbamV7.py — 推理阶段 checkpoint 信息输出逻辑。
- d:/Work space/DeepLearning/farm/crop_segmentation/core/models_dl.py — 1D-CNN 分类基线（可作为补充背景，不纳入本文主线实验）。
- d:/Work space/DeepLearning/farm/crop_segmentation/interfaces/train_interface.py — 分类任务评估口径示例（accuracy + classification_report）。

**Verification**
1. 逐条校对论文中的方法声明与源码对应关系，至少覆盖模型结构、损失、验证指标、早停与保存策略。
2. 确认“当前可确认精度”仅引用代码中明确字段（best_iou、val_ious、IoU 计算逻辑），不臆造具体分数。
3. 补测后对比三类结果：像素级指标（IoU/Dice/F1）、边界质量指标（Boundary F1/HD95 可选）、矢量级质量（面积误差/拓扑错误率）。
4. 对所有改进实验记录同一数据划分、同一阈值策略、同一后处理开关，保证可复现与公平对比。

**Decisions**
- 仅纳入耕地分割深度学习主线，不展开分类模型主实验。
- 当前阶段先给“代码可确认口径”，具体数值在后续补测中填充。
- 优先改进方向：结构改进、损失与训练策略、后处理与矢量化。

**Further Considerations**
1. 多光谱通道接入方案优先级：Option A 直接改第一层输入通道；Option B 双分支（RGB + 光谱）后融合；Option C 先做波段选择再扩通道。推荐 A 作为首轮低风险基线。
2. 消融规模控制：Option A 小规模快速筛选（每组 20-30 epoch）；Option B 全量训练（50+ epoch）。推荐先 A 后 B。
3. 论文定位：Option A 工程落地导向（强调自动矢量化流程）；Option B 方法改进导向（强调多任务与注意力贡献）。推荐 A+B 结合。