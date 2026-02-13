# Offline cached solvability pseudo-trust + trust-head-only training.
#
# Workflow:
# 1) Generate cache once:
#    python projects/self_verify_pose/tools/generate_solvability_cache.py \
#      projects/self_verify_pose/configs/baseline_rtmpose_s_1x32g_coco-256x192.py \
#      /path/to/pose_ckpt.pth --split train --out work_dirs/self_verify_pose/cache/train_sol.npz
# 2) Train trust head using cache:
#    PSEUDO_TRUST_CACHE=work_dirs/self_verify_pose/cache/train_sol.npz \
#      POSE_CKPT=/path/to/pose_ckpt.pth \
#      mim train mmpose projects/self_verify_pose/configs/solvability_cached_trusthead_rtmpose_s_1x32g_coco-256x192.py

import os

_base_ = './baseline_rtmpose_s_1x32g_coco-256x192.py'

# Dataset root (override via env if needed)
coco_root = os.environ.get('COCO_ROOT', '/root/autodl-tmp/coco/')

custom_imports = dict(
    imports=[
        'projects.self_verify_pose.models',
        'projects.self_verify_pose.metrics',
        'projects.self_verify_pose.datasets',
    ],
    allow_failed_imports=False,
)

# Load a fully-trained pose model (not just backbone init)
pose_ckpt = os.environ.get('POSE_CKPT', '')
if pose_ckpt:
    load_from = pose_ckpt

cache_file = os.environ.get('PSEUDO_TRUST_CACHE', '')
if not cache_file:
    # Keep config runnable, but user should set env var.
    cache_file = 'work_dirs/self_verify_pose/cache/train_sol.npz'

model = dict(
    type='SelfVerifyTopdownPoseEstimator',
    full_param_train=False,
    trust_head=dict(type='TrustHead', in_channels=512, num_keypoints=17),
    trust_cfg=dict(
        enable=True,
        source='solvability_cached',
        pose_loss=False,
        loss_weight=1.0,
        detach_pseudo=True,
    ),
    consistency_cfg=dict(enable=False),
    solvability_cfg=dict(enable=False),
)

# Use a deterministic pipeline (val_pipeline) + cached pseudo labels.
# We don't need GenerateTarget since pose loss is disabled.
backend_args = dict(backend='local')
cache_train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(192, 256)),
    dict(type='LoadPseudoTrust', cache_file=cache_file, field_name='pseudo_trust_sol', missing='error'),
    dict(type='PackPoseInputs'),
]
train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        pipeline=cache_train_pipeline
    ),
)

_smoke_indices = os.environ.get('SMOKE_INDICES', '').strip()
if _smoke_indices:
    _n = int(_smoke_indices)
    # mmengine BaseDataset supports `indices=int` to take the first N samples.
    train_dataloader['dataset']['indices'] = _n
    # Also speed up val/test for smoke runs.
    val_dataloader = dict(dataset=dict(indices=_n))
    test_dataloader = val_dataloader

max_epochs = 10
train_cfg = dict(max_epochs=max_epochs, val_interval=1)

# Rebuild schedulers so they match the shortened max_epochs
base_lr = 1.0e-3
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=200),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=0,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

val_evaluator = [
    dict(type='CocoMetric', ann_file=coco_root + 'annotations/person_keypoints_val2017.json'),
    dict(type='JointTrustMetric', dist_thr=0.05, num_bins=15),
]

test_evaluator = val_evaluator

work_dir = 'work_dirs/self_verify_pose/solvability_cached_trusthead'

_use_wandb = os.environ.get('USE_WANDB', '0') == '1'
if _use_wandb:
    vis_backends = [
        dict(type='LocalVisBackend'),
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(
                project=os.environ.get('WANDB_PROJECT', 'self_verify_pose'),
                name=os.environ.get('WANDB_NAME', 'offline_cached_trusthead_rtmpose_s'),
            ),
        ),
    ]
    visualizer = dict(type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')
