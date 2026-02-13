# Deterministic dataloader config for generating solvability cache.
# Uses the same COCO root as the baseline config and swaps train pipeline -> val_pipeline
# to avoid random augmentations during cache generation.

_base_ = './baseline_rtmpose_s_1x32g_coco-256x192.py'

# NOTE: The derived config file cannot directly reference variables defined in
# `_base_` at parse time. We define a deterministic, val-style pipeline here.
backend_args = dict(backend='local')
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(192, 256)),
    dict(type='PackPoseInputs'),
]

# Make iteration order stable (not required, but nice for debugging)
train_dataloader = dict(
    sampler=dict(shuffle=False),
    dataset=dict(
        pipeline=val_pipeline,
    ),
)
