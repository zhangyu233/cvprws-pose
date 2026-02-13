# 指标说明（Self-Verify Pose）

这份文档解释训练/验证过程中日志（终端、`vis_data/*.json`、W&B）里常见的指标名称：它们来自哪个阶段（train/val/test）、对应什么集合（COCO train/val）、以及大致的计算含义。

## 1. 指标从哪里来

在 MMPose/MMEngine 中：

- **训练阶段（train）**：Runner 每隔 `default_hooks.logger.interval` 个 iteration，会把 `model.loss()` 返回的标量写入日志。
- **验证阶段（val）/测试阶段（test）**：Runner 在 `val_interval` 触发评测时，调用 `model.predict()` 产出预测，并交给 `val_evaluator/test_evaluator` 计算指标（例如 COCO AP、trust 指标）。

因此你会看到两类指标：

- **train（迭代级别）**：更频繁，用于看 loss 是否下降、训练是否稳定。
- **val/test（epoch 级别）**：较少，但更“真实”，例如 COCO AP。

> 小提示：在 W&B 面板里，如果你把横轴用 `step`（全局 step）并叠加 `coco/AP`，它会呈现为“每个 epoch 一个点”的阶梯状曲线，这是正常现象。

## 2. 训练阶段（train）常见指标

这些指标通常出现在 `Epoch(train) [...]` 的日志行，或 `vis_data/<timestamp>.json` 的行级 JSON 里。

- **loss**：训练总损失（MMEngine 会把所有 `loss_*` 项按权重求和得到）。
- **loss_kpt**：原始姿态估计的关键点监督损失（RTMPose-S 的主任务 loss）。
- **acc_pose**：训练 batch 上的一个快速 proxy 指标（用于监控训练是否“在动”），不是 COCO AP。

### 2.1 Self-Verify 相关（你新增的）

以下项来自 `SelfVerifyTopdownPoseEstimator.loss()`：

- **loss_consistency_rot**：2D 旋转等变一致性损失。做法是对输入 crop 随机旋转得到第二视角，约束“旋转前预测经几何变换后”与“旋转后直接预测”一致。
- **rot_inconsistency_mean**：旋转一致性误差的平均值（按 joint 聚合）。数值越小表示越一致。
- **pseudo_trust_mean**：pseudo-trust 的平均值。pseudo-trust 通常由 `exp(-alpha * inconsistency)` 得到（不一致越大，可信度越低）。
- **loss_trust**：trust head 的监督损失（通常为 `BCE(trust_pred, pseudo_trust)`）。
- **trust_pred_mean**：trust head 输出的平均值（0~1）。

### 2.2 优化/性能相关

- **lr / base_lr**：当前学习率 / 基准学习率（有 warmup 或 scheduler 时两者可能不同）。
- **time / data_time**：迭代耗时 / 数据加载耗时。
- **memory**：显存占用（MB）。

## 3. 验证/测试阶段（val/test）COCO 指标

这些指标来自 `CocoMetric`，在 COCO val2017 上计算（top-down 使用 GT bbox）。常见字段：

- **coco/AP**：主指标（OKS mAP，0.50:0.05:0.95 平均）。
- **coco/AP .5**：OKS=0.5 下的 AP。
- **coco/AP .75**：OKS=0.75 下的 AP。
- **coco/AP (M)**、**coco/AP (L)**：中等/大目标尺度分组 AP。
- **coco/AR** 及其变体：对应的 Average Recall。

## 4. 验证/测试阶段（val/test）Trust 指标

这些指标来自 `JointTrustMetric`，前提是模型在 `predict()` 时把 per-joint trust 写入 `pred_instances.keypoint_trust`。

### 4.1 correctness label（什么是“对/错”）

对每个 joint，定义二分类标签：

- 设预测点为 $\hat{p}$，GT 点为 $p$，误差 $d=\|\hat{p}-p\|_2$
- 阈值为 `dist_thr * max(bbox_w, bbox_h)`
- 若 $d$ 小于阈值，则该 joint **correct=1**，否则 **correct=0**

并且通常会用 `keypoints_visible` 过滤不可见点。

### 4.2 指标含义

- **trust/AUROC**：把 trust 当作区分 correct/incorrect 的 score，计算 ROC AUC。
- **trust/AUPR**：Average Precision（PR 曲线面积），在正样本稀疏时更敏感。
- **trust/ECE**：Expected Calibration Error。衡量“预测置信度”与“真实正确率”的偏差。
- **trust/Acc@0.5**：用阈值 0.5 把 trust 二值化后，与 correct 标签比对得到的 accuracy。
- **trust/PosRate**：correct=1 的比例（数据分布），用于辅助解读 AUROC/AUPR。

## 5. 你现在在 “test” 上到底测了哪些指标？

在当前项目配置里（`test_evaluator = val_evaluator`），`mim test` 默认是在 **COCO val2017** 上做最终评测留档（不是 COCO test-dev）。因此 test 阶段会输出两大类指标：

### 5.1 姿态指标（来自 CocoMetric）

你会在终端看到这些 key（并写入 `vis_data/*.json` / W&B）：

- `coco/AP`
- `coco/AP .5`
- `coco/AP .75`
- `coco/AP (M)`, `coco/AP (L)`
- `coco/AR`
- `coco/AR .5`
- `coco/AR .75`
- `coco/AR (M)`, `coco/AR (L)`

### 5.2 可信度指标（来自 JointTrustMetric）

只要模型在 `predict()` 时写出了 `pred_instances.keypoint_trust`，test 阶段会额外输出：

- `trust/AUROC`
- `trust/AUPR`
- `trust/ECE`
- `trust/Acc@0.5`
- `trust/PosRate`

这些 trust 指标同样会在 `mim test` 的最终汇总行中出现，例如：

`Epoch(test) ... coco/AP: ... trust/AUROC: ... trust/AUPR: ...`

## 6. 如何在 W&B 面板里更好地看

- **train 曲线**：用 `loss / loss_kpt / loss_consistency_rot / loss_trust` 看训练是否稳定。
- **val 点**：用 `coco/AP`（以及 `trust/*`）看真正的效果变化。
- 如果要比较多组实验，建议固定：
  - `WANDB_PROJECT` 同一个
  - `WANDB_NAME` 写入关键超参（例如 `rot30_alpha10_w1_trust1_bs16`）

