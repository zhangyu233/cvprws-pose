# Cycle-only self-verification skeleton (placeholder).
# For now it only switches model type to a wrapper, baseline losses unchanged.

_base_ = './baseline_rtmpose_s_1x32g_coco-256x192.py'

custom_imports = dict(imports='models', allow_failed_imports=False)

# Only override model type; keep backbone/head/codec etc from baseline.
model = dict(
    type='SelfVerifyTopdownPoseEstimator',
    trust_head=dict(type='TrustHead', in_channels=512, num_keypoints=17),
    trust_cfg=dict(
        enable=False,  # set True after implementing cycle trust loss
        alpha=10.0,
        loss_weight=1.0,
    ),
)

work_dir = 'work_dirs/self_verify_pose/selfverify_cycle'
