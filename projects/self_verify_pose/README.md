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

说明（很重要）：本项目里默认的 “test” 概念是 **用某个 checkpoint 在 COCO val2017 上做一次最终评测留档**。

- 目前 `test_evaluator = val_evaluator`，因此 test 阶段计算的也是 val2017 的指标（例如 `coco/AP`）。
- top-down 默认使用 GT bbox（配置里未设置 `bbox_file`）。

如果你后续真的要做“测试集”（例如 COCO test-dev），需要：
- 更换 `test_dataloader.dataset.ann_file / data_prefix`（或使用官方对应 config），并把 `test_evaluator.ann_file` 指向对应的测试标注/服务。

提示：如果 `latest.pth` 不存在，可以改用 `epoch_5.pth` 或 `best_coco_AP*.pth`（都在 work_dir 里）。

## Self-Verify (Rotation Consistency + Trust)

该配置会：
- 在训练中加入 2D rotation equivariance consistency loss
- 同时训练 joint-level trust head（可在 config 里开关）
- 在验证时额外输出 trust 指标（AUROC/AUPR/ECE 等）

当前 self-verify 配置文件：
- configs/selfverify_cycle_rtmpose_s_1x32g_coco-256x192.py

Solvability-only（只实现“可解性” teacher 的版本）配置文件：
- configs/solvability_rtmpose_s_1x32g_coco-256x192.py

Offline cached solvability（离线缓存伪标签，只训练 trust head）配置文件：
- configs/solvability_cached_trusthead_rtmpose_s_1x32g_coco-256x192.py

离线缓存的生成配置（用于避免数据增强带来的随机性）：
- configs/cachegen_rtmpose_s_1x32g_coco-256x192.py

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

Solvability-only 训练：
```bash
mim train mmpose projects/self_verify_pose/configs/solvability_rtmpose_s_1x32g_coco-256x192.py \
  --work-dir work_dirs/self_verify_pose/solvability
```

### Offline cached solvability → Train trust head only

目标：
- 用一个“已经训练好的 pose checkpoint”离线跑一遍 solvability teacher，把每个样本的 `pseudo_trust_sol` 缓存到 `.npz`
- 训练阶段不再在线跑 teacher，只读取缓存并冻结 pose（只更新 trust_head）

一键脚本（推荐）：
```bash
bash projects/self_verify_pose/run_offline_cached_trusthead.sh /path/to/pose_ckpt.pth
```

说明：如果你本地已有 `checkpoints/rtmpose-s_simcc-coco_256x192.pth`（本仓库常用），脚本也支持不传参数直接跑，会自动使用该 checkpoint。

可选（带 W&B 记录）：
```bash
WANDB_PROJECT=self_verify_pose \
WANDB_NAME=offline_cached_trusthead \
bash projects/self_verify_pose/run_offline_cached_trusthead_wandb.sh /path/to/pose_ckpt.pth
```

FK 版本（用 `pytorch_kinematics` 的显式骨架 FK + 关节角 IK 来生成伪标签；更贴近论文里的“IK inverse problem”表述）：
```bash
bash projects/self_verify_pose/run_offline_cached_trusthead_fk.sh /path/to/pose_ckpt.pth
```

说明：同样支持不传参数时自动使用 `checkpoints/rtmpose-s_simcc-coco_256x192.pth`（若存在）。

说明：FK 版本需要额外依赖 `pytorch-kinematics`（脚本会自动检测并尝试安装）。

脚本会默认把缓存写到：
- free teacher：`work_dirs/self_verify_pose/cache/train_sol.npz`
- FK teacher：`work_dirs/self_verify_pose/cache/train_sol_fk.npz`

如果你设置了 `MAX_SAMPLES=N` 做 smoke，脚本会改用单独的缓存文件名，避免覆盖 full cache：
- free teacher：`work_dirs/self_verify_pose/cache/train_sol_N.npz`
- FK teacher：`work_dirs/self_verify_pose/cache/train_sol_fk_N.npz`

常用环境变量：
- `CACHE_TRAIN=...` 指定缓存文件路径
- `FORCE_CACHE=1` 强制重新生成缓存
- `MAX_SAMPLES=1000` 快速 smoke（同时限制 cache + 训练数据集为前 N 个样本，避免“cache 里找不到 id”的报错）
- `WORK_DIR=...` 指定训练 work_dir

如果你想手动分两步跑：
1) 生成缓存：
```bash
python projects/self_verify_pose/tools/generate_solvability_cache.py \
  projects/self_verify_pose/configs/cachegen_rtmpose_s_1x32g_coco-256x192.py \
  /path/to/pose_ckpt.pth \
  --split train \
  --out work_dirs/self_verify_pose/cache/train_sol.npz \
  --use-gt-visible
```

2) 训练 trust head（冻结 pose，只用缓存监督）：
```bash
PSEUDO_TRUST_CACHE=work_dirs/self_verify_pose/cache/train_sol.npz \
POSE_CKPT=/path/to/pose_ckpt.pth \
mim train mmpose projects/self_verify_pose/configs/solvability_cached_trusthead_rtmpose_s_1x32g_coco-256x192.py \
  --work-dir work_dirs/self_verify_pose/solvability_cached_trusthead
```

或直接用脚本（推荐，开箱即跑）：
```bash
bash projects/self_verify_pose/run_train_selfverify.sh
```

### Monitoring (训练过程看指标)

指标字段含义说明见：[projects/self_verify_pose/docs/metrics_guide.md](projects/self_verify_pose/docs/metrics_guide.md)

训练过程中主要看两类输出：

1) 终端/日志文件（实时）：
- 日志文件位置：`work_dirs/self_verify_pose/selfverify_cycle/<timestamp>/<timestamp>.log`
- 实时查看：
```bash
tail -f work_dirs/self_verify_pose/selfverify_cycle/*/*.log
```

2) 可画曲线的 JSON 标量日志（推荐做曲线对比）：
- 文件位置：`work_dirs/.../<timestamp>/vis_data/<timestamp>.json`（每行一个 JSON，包含 loss/lr 等标量）
- 画训练曲线（示例：loss + consistency + trust）：
```bash
python tools/analysis_tools/analyze_logs.py plot_curve \
  work_dirs/self_verify_pose/selfverify_cycle/*/vis_data/*.json \
  --keys loss loss_kpt loss_consistency_rot loss_trust rot_inconsistency_mean trust_pred_mean \
  --out work_dirs/self_verify_pose/selfverify_cycle/train_curves.png
```

常用的快速覆盖（不改文件，直接命令行调参）：

- 一条命令同时设置 batch_size / dataloader workers / persistent_workers / AMP / 梯度累积（常用于控显存 + 控速度）：
```bash
mim train mmpose projects/self_verify_pose/configs/selfverify_cycle_rtmpose_s_1x32g_coco-256x192.py \
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
mim train mmpose projects/self_verify_pose/configs/selfverify_cycle_rtmpose_s_1x32g_coco-256x192.py \
  --cfg-options model.trust_cfg.enable=False
```

- 只训 trust（不加 consistency；一般不建议，但你可以做 ablation）：
```bash
mim train mmpose projects/self_verify_pose/configs/selfverify_cycle_rtmpose_s_1x32g_coco-256x192.py \
  --cfg-options model.consistency_cfg.enable=False
```

- 冻结 backbone/neck/head（仅训练 trust_head；用于快速看 trust 是否能学到信号）：
```bash
mim train mmpose projects/self_verify_pose/configs/selfverify_cycle_rtmpose_s_1x32g_coco-256x192.py \
  --cfg-options model.full_param_train=False
```

- 调大旋转范围 / 调整 loss 权重：
```bash
mim train mmpose projects/self_verify_pose/configs/selfverify_cycle_rtmpose_s_1x32g_coco-256x192.py \
  --cfg-options model.consistency_cfg.max_deg=45.0 model.consistency_cfg.loss_weight=0.5
```

### W&B (wandb)

如果你想在网页端实时看训练曲线（loss/lr/自定义 consistency+trust 指标），可以用 wandb：

1) 安装并登录：
```bash
pip install -U wandb
wandb login
```

2) 用环境变量打开 W&B（不需要额外的 wandb 配置文件）：
```bash
USE_WANDB=1 \
WANDB_PROJECT=self_verify_pose \
WANDB_NAME=selfverify_cycle_rtmpose_s \
mim train mmpose projects/self_verify_pose/configs/selfverify_cycle_rtmpose_s_1x32g_coco-256x192.py \
  --work-dir work_dirs/self_verify_pose/selfverify_cycle
```

可选：离线模式（不上传，只落本地文件，之后可 `wandb sync`）：
```bash
USE_WANDB=1 WANDB_MODE=offline \
mim train mmpose projects/self_verify_pose/configs/selfverify_cycle_rtmpose_s_1x32g_coco-256x192.py \
  --work-dir work_dirs/self_verify_pose/selfverify_cycle
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

### 记录测试集效果（如何留档）

在 MMEngine/MMPose 里，`mim test` 会把指标打印到终端，同时把日志和可画曲线的标量落到 `--work-dir` 下。

在本项目的 self-verify 配置中，test 时除了 `coco/*` 之外还会输出 `trust/*`（例如 `trust/AUROC`, `trust/AUPR`, `trust/ECE`），用于评估 trust head 对“正确/错误 joint”的区分与校准能力。

- 推荐给每次评测一个独立 work-dir（便于对比/回溯）：
```bash
mim test mmpose projects/self_verify_pose/configs/selfverify_cycle_rtmpose_s_1x32g_coco-256x192.py \
  --checkpoint work_dirs/self_verify_pose/selfverify_cycle/best_coco_AP_epoch_5.pth \
  --work-dir work_dirs/self_verify_pose/selfverify_cycle/test_best_epoch5
```

- 如果你想把 test 指标也同步到 W&B（作为一个单独 run 记录）：
```bash
WANDB_PROJECT=self_verify_pose WANDB_NAME=selfverify_cycle_test_best_epoch5 \
bash projects/self_verify_pose/run_test_selfverify_wandb.sh \
  work_dirs/self_verify_pose/selfverify_cycle/best_coco_AP_epoch_5.pth
```

说明：因为 `test_evaluator = val_evaluator`，这里的 “test” 仍然是在 COCO val2017 上做最终评测留档（不是 COCO test-dev）。

提示：实际 ckpt 路径通常为 work_dir 下的 EXP_NAME 子目录（例如 epoch_1.pth / latest.pth / best_coco_AP*.pth）。
