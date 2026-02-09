# Single-GPU (32GB) friendly baseline config.
# Inherits official RTMPose-S COCO top-down recipe (GT bbox by default).

_base_ = 'mmpose::body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192.py'

# ---- Data root (your COCO path, GT bbox) ----
coco_root = '/root/autodl-tmp/coco/'

# evaluators (must use absolute COCO path)
val_evaluator = dict(
    type='CocoMetric',
    ann_file=coco_root + 'annotations/person_keypoints_val2017.json')
test_evaluator = val_evaluator

# ---- Runtime: quick sanity run (increase later) ----
max_epochs = 5
train_cfg = dict(max_epochs=max_epochs, val_interval=1)

# ---- Single GPU batch + LR (linear scaling from 1024 -> 32) ----
train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        data_root=coco_root,
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/')))
val_dataloader = dict(
    batch_size=64,
    num_workers=8,
    dataset=dict(
        data_root=coco_root,
        ann_file='annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='val2017/')))
test_dataloader = val_dataloader

base_lr = 1.25e-4
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# Rebuild schedulers so they match the shortened max_epochs
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=0,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
]

# Hooks: update PipelineSwitchHook epoch so it matches our max_epochs
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49)
]

default_hooks = dict(
    checkpoint=dict(
        save_best='coco/AP',
        rule='greater',
        max_keep_ckpts=2,
        interval=1))

work_dir = 'work_dirs/self_verify_pose/baseline'
