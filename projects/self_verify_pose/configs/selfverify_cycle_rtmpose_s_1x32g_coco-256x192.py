# Cycle-only self-verification skeleton (placeholder).
# For now it only switches model type to a wrapper, baseline losses unchanged.

_base_ = './baseline_rtmpose_s_1x32g_coco-256x192.py'

custom_imports = dict(imports=['models', 'metrics'], allow_failed_imports=False)

# Only override model type; keep backbone/head/codec etc from baseline.
model = dict(
    type='SelfVerifyTopdownPoseEstimator',
    trust_head=dict(type='TrustHead', in_channels=512, num_keypoints=17),
    trust_cfg=dict(
        enable=True,
        loss_weight=1.0,
        detach_pseudo=True,
    ),
    consistency_cfg=dict(
        enable=True,
        max_deg=30.0,
        alpha=10.0,
        loss_weight=1.0,
        use_gt_visible=True,
        detach_pseudo=True,
    ),
)

# Keep COCO AP evaluator and add joint-trust metric
val_evaluator = [
    dict(type='CocoMetric', ann_file='/root/autodl-tmp/coco/annotations/person_keypoints_val2017.json'),
    dict(type='JointTrustMetric', dist_thr=0.05, num_bins=15, prefix='trust'),
]
test_evaluator = val_evaluator

work_dir = 'work_dirs/self_verify_pose/selfverify_cycle'
