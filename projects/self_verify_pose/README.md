# Self-Verify Pose (Project Skeleton)

本项目用于在 MMPose 上复现/开发「自验证 + 关键点可信度（trust）」相关实验。

## Prerequisites
- 建议使用 VS Code Remote-SSH 直接在远程机器运行
- 该 baseline 继承 RTMPose-S（COCO, top-down）。其 backbone 来自 `mmdet` 的 CSPNeXt，如环境缺依赖需安装 mmdet。

## Data (COCO, GT box)
当前实验配置使用的 COCO 根目录为：

- `/root/autodl-tmp/coco/`

其结构应为：

```
/root/autodl-tmp/coco/
  train2017/
  val2017/
  annotations/
    person_keypoints_train2017.json
    person_keypoints_val2017.json
```

说明：本配置不设置 `bbox_file`，因此 top-down 评测默认使用 GT bbox。

## Usage
推荐直接从 MMPose 仓库根目录运行（不需要额外设置 PYTHONPATH）：

```bash
cd /code/mmpose
```

说明：self-verify 配置已使用 `custom_imports=['projects.self_verify_pose.models', 'projects.self_verify_pose.metrics']`，所以从仓库根目录运行即可正常 import 到自定义模块。

### Train (single GPU, 32GB friendly)
```bash
mim train mmpose projects/self_verify_pose/configs/baseline_rtmpose_s_1x32g_coco-256x192.py
```

说明：该 baseline 配置目前是 5 epoch 的快速 sanity run（每个 epoch 都会 val，并保存 checkpoint）。

baseline 关键可调参数（见 configs/baseline_rtmpose_s_1x32g_coco-256x192.py）：
- 训练：batch_size=32，max_epochs=5，val_interval=1
- 优化：AdamW，lr=1.25e-4
- 保存：每个 epoch 保存 ckpt，save_best=coco/AP，max_keep_ckpts=2

### Validate (pretrained checkpoint, no training)
先用官方 RTMPose-S 权重快速验证数据/评测链路：

```bash
mim test mmpose projects/self_verify_pose/configs/baseline_rtmpose_s_1x32g_coco-256x192.py \
  --checkpoint /code/mmpose/checkpoints/rtmpose-s_simcc-coco_256x192.pth \
  --work-dir work_dirs/self_verify_pose/validate_pretrained
```

### Test
```bash
mim test mmpose projects/self_verify_pose/configs/baseline_rtmpose_s_1x32g_coco-256x192.py \
  --checkpoint work_dirs/self_verify_pose/baseline_quick/EXP_NAME/latest.pth
```

提示：如果 `latest.pth` 不存在，可以改用 `epoch_5.pth` 或 `best_coco_AP*.pth`（都在 work_dir 里）。

## Self-Verify (Rotation Consistency + Trust)

该配置会：
- 在训练中加入 2D rotation equivariance consistency loss
- 同时训练 joint-level trust head（可在 config 里开关）
- 在验证时额外输出 trust 指标（AUROC/AUPR/ECE 等）

当前 self-verify 配置文件：
- configs/selfverify_cycle_rtmpose_s_1x32g_coco-256x192.py

self-verify 关键可调参数（在 config 的 model 字段里）：

- 全参训练控制：
  - model.full_param_train：是否全参 finetune（默认 True）。设为 False 时会冻结 backbone/neck/head，仅训练 trust_head（仍会正常 forward 计算 pose 与 consistency 的信号，但不回传这些模块的梯度）。

- 一致性损失（rotation equivariance）：
  - model.consistency_cfg.enable：是否启用（默认 True）
  - model.consistency_cfg.max_deg：随机旋转角度范围（默认 30 度）
  - model.consistency_cfg.alpha：把不一致性映射为 pseudo-trust 的系数（默认 10）
  - model.consistency_cfg.loss_weight：loss_consistency_rot 权重（默认 1）
  - model.consistency_cfg.use_gt_visible：是否用 GT visibility mask（默认 True）

- trust head：
  - model.trust_cfg.enable：是否训练 trust（默认 True）
  - model.trust_cfg.loss_weight：loss_trust 权重（默认 1）
  - model.trust_cfg.detach_pseudo：pseudo-trust 是否 stop-gradient（默认 True）

- trust 指标（验证时输出）：
  - JointTrustMetric.dist_thr：正确/错误阈值（默认 0.05 * max(bbox_w, bbox_h)）
  - JointTrustMetric.num_bins：ECE 分桶数（默认 15）

训练：
```bash
mim train mmpose projects/self_verify_pose/configs/selfverify_cycle_rtmpose_s_1x32g_coco-256x192.py \
  --work-dir work_dirs/self_verify_pose/selfverify_cycle
```

或直接用脚本（推荐，开箱即跑）：
```bash
bash projects/self_verify_pose/run_train_selfverify.sh
```

常用的快速覆盖（不改文件，直接命令行调参）：

- 一条命令同时设置 batch_size / dataloader workers / persistent_workers / AMP / 梯度累积（常用于控显存 + 控速度）：
```bash
mim train mmpose configs/selfverify_cycle_rtmpose_s_1x32g_coco-256x192.py \
  --cfg-options \
  train_dataloader.batch_size=16 \
  train_dataloader.num_workers=4 \
  train_dataloader.persistent_workers=True \
  optim_wrapper.type=AmpOptimWrapper \
  optim_wrapper.loss_scale=dynamic \
  optim_wrapper.accumulative_counts=2
```

提示：`persistent_workers=True` 需要 `num_workers>0`；如果你想用 `num_workers=0`，就把 `train_dataloader.persistent_workers=False`。

- 只训 consistency，不训 trust：
```bash
mim train mmpose configs/selfverify_cycle_rtmpose_s_1x32g_coco-256x192.py \
  --cfg-options model.trust_cfg.enable=False
```

- 只训 trust（不加 consistency；一般不建议，但你可以做 ablation）：
```bash
mim train mmpose configs/selfverify_cycle_rtmpose_s_1x32g_coco-256x192.py \
  --cfg-options model.consistency_cfg.enable=False
```

- 冻结 backbone/neck/head（仅训练 trust_head；用于快速看 trust 是否能学到信号）：
```bash
mim train mmpose configs/selfverify_cycle_rtmpose_s_1x32g_coco-256x192.py \
  --cfg-options model.full_param_train=False
```

- 调大旋转范围 / 调整 loss 权重：
```bash
mim train mmpose configs/selfverify_cycle_rtmpose_s_1x32g_coco-256x192.py \
  --cfg-options model.consistency_cfg.max_deg=45.0 model.consistency_cfg.loss_weight=0.5
```

测试（COCO AP + trust metrics）：
```bash
mim test mmpose projects/self_verify_pose/configs/selfverify_cycle_rtmpose_s_1x32g_coco-256x192.py \
  --checkpoint work_dirs/self_verify_pose/selfverify_cycle/latest.pth
```

或用脚本：
```bash
bash projects/self_verify_pose/run_test_selfverify.sh work_dirs/self_verify_pose/selfverify_cycle/latest.pth
```

提示：实际 ckpt 路径通常为 work_dir 下的 EXP_NAME 子目录（例如 epoch_1.pth / latest.pth / best_coco_AP*.pth）。
