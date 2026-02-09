# Self-Verify Pose (Project Skeleton)

本项目用于在 MMPose 上复现/开发「自验证 + 关键点可信度（trust）」相关实验。

## Prerequisites
- 建议使用 VS Code Remote-SSH 直接在远程机器运行
- 该 baseline 继承 RTMPose-S（COCO, top-down）。其 backbone 来自 `mmdet` 的 CSPNeXt，如环境缺依赖需安装 mmdet。

## Data (COCO, GT box)
按 MMPose 约定放置：

```
data/coco/
  train2017/
  val2017/
  annotations/
    person_keypoints_train2017.json
    person_keypoints_val2017.json
```

说明：本配置不设置 `bbox_file`，因此 top-down 评测默认使用 GT bbox。

## Usage
在本项目根目录执行（确保能 import 到 `models/`）：

```bash
cd projects/self_verify_pose
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### Train (single GPU, 32GB friendly)
```bash
mim train mmpose configs/baseline_rtmpose_s_1x32g_coco-256x192.py
```

说明：该 baseline 配置目前是 5 epoch 的快速 sanity run（每个 epoch 都会 val，并保存 checkpoint）。

### Validate (pretrained checkpoint, no training)
先用官方 RTMPose-S 权重快速验证数据/评测链路：

```bash
mim test mmpose configs/baseline_rtmpose_s_1x32g_coco-256x192.py \
  --checkpoint /code/mmpose/checkpoints/rtmpose-s_simcc-coco_256x192.pth \
  --work-dir work_dirs/self_verify_pose/validate_pretrained
```

### Test
```bash
mim test mmpose configs/baseline_rtmpose_s_1x32g_coco-256x192.py \
  work_dirs/self_verify_pose/baseline/latest.pth
```

提示：如果 `latest.pth` 不存在，可以改用 `epoch_5.pth` 或 `best_coco_AP*.pth`（都在 work_dir 里）。

## Next
- `configs/selfverify_cycle_rtmpose_s_1x32g_coco-256x192.py` 目前是“空壳入口”，后续会在 `models/selfverify_wrapper.py` 的 `loss()` 中加入 cycle-only 自验证训练逻辑。
