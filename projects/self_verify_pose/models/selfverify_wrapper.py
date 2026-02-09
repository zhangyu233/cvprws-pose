from __future__ import annotations

from typing import Optional

from torch import Tensor

from mmpose.models.pose_estimators.topdown import TopdownPoseEstimator
from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType, OptConfigType, OptMultiConfig, SampleList


@MODELS.register_module()
class SelfVerifyTopdownPoseEstimator(TopdownPoseEstimator):
    """Top-down estimator wrapper for self-verification experiments (skeleton).

    Current behavior: identical to TopdownPoseEstimator.
    Next step: add cycle-consistency / pseudo-view / IK pseudo labels and
    supervise a trust head.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 metainfo: Optional[dict] = None,
                 trust_head: Optional[dict] = None,
                 trust_cfg: Optional[dict] = None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            metainfo=metainfo)

        self.trust_cfg = trust_cfg or {}
        self.with_trust = trust_head is not None
        self.trust_head = MODELS.build(trust_head) if self.with_trust else None

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        losses = super().loss(inputs, data_samples)

        # Placeholder: add trust loss here later.
        # if self.with_trust and self.trust_cfg.get('enable', False):
        #     feats = self.extract_feat(inputs)
        #     feat = feats[0] if isinstance(feats, (list, tuple)) else feats
        #     t = self.trust_head(feat)
        #     losses['loss_trust'] = t.mean() * 0.0

        return losses
