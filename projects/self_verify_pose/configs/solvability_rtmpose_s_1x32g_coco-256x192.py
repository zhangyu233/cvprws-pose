# Solvability-only self-verification (IK-style teacher) + trust head.
# This config disables rotation-consistency and uses the solvability teacher
# to generate pseudo-trust labels.

import os

_base_ = './baseline_rtmpose_s_1x32g_coco-256x192.py'

custom_imports = dict(
    imports=[
        'projects.self_verify_pose.models',
        'projects.self_verify_pose.metrics',
    ],
    allow_failed_imports=False)

model = dict(
    type='SelfVerifyTopdownPoseEstimator',
    trust_head=dict(type='TrustHead', in_channels=512, num_keypoints=17),
    trust_cfg=dict(
        enable=True,
        loss_weight=1.0,
        detach_pseudo=True,
        source='solvability',
    ),
    # Disable rotation-consistency for this baseline
    consistency_cfg=dict(enable=False),
    # Enable solvability teacher
    solvability_cfg=dict(
        enable=True,
        num_iters=15,
        lr=0.05,
        w_reproj=1.0,
        w_bone=10.0,
        w_angle=1.0,
        w_reg=1e-3,
        alpha=10.0,
        use_gt_visible=True,
        detach=True,
    ),
)

val_evaluator = [
    dict(type='CocoMetric', ann_file='/root/autodl-tmp/coco/annotations/person_keypoints_val2017.json'),
    dict(type='JointTrustMetric', dist_thr=0.05, num_bins=15),
]

test_evaluator = val_evaluator

work_dir = 'work_dirs/self_verify_pose/solvability'

_use_wandb = os.environ.get('USE_WANDB', '0') == '1'
if _use_wandb:
    vis_backends = [
        dict(type='LocalVisBackend'),
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(
                project=os.environ.get('WANDB_PROJECT', 'self_verify_pose'),
                name=os.environ.get('WANDB_NAME', 'solvability_rtmpose_s'),
            ),
        ),
    ]
    visualizer = dict(type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')
