from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor

from mmpose.models.pose_estimators.topdown import TopdownPoseEstimator
from mmpose.registry import MODELS
from mmpose.utils.typing import (ConfigType, OptConfigType, OptMultiConfig,
                                 SampleList)


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
                 trust_cfg: Optional[dict] = None,
                 consistency_cfg: Optional[dict] = None,
                 full_param_train: bool = True):
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
        self.consistency_cfg = consistency_cfg or {}
        self.with_trust = trust_head is not None
        self.trust_head = MODELS.build(trust_head) if self.with_trust else None

        # Training control: by default we finetune the whole model.
        # If set to False, we freeze backbone/neck/head and only train trust_head.
        self.full_param_train = bool(full_param_train)
        if not self.full_param_train:
            self._freeze_module_params(self.backbone)
            if getattr(self, 'neck', None) is not None:
                self._freeze_module_params(self.neck)
            if getattr(self, 'head', None) is not None:
                self._freeze_module_params(self.head)
            # trust_head remains trainable when present.

    @staticmethod
    def _freeze_module_params(module) -> None:
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen parts in eval mode to avoid BN running-stat updates.
        if mode and not getattr(self, 'full_param_train', True):
            if getattr(self, 'backbone', None) is not None:
                self.backbone.eval()
            if getattr(self, 'neck', None) is not None:
                self.neck.eval()
            if getattr(self, 'head', None) is not None:
                self.head.eval()
            if getattr(self, 'trust_head', None) is not None:
                self.trust_head.train()
        return self

    @staticmethod
    def _as_last_feat(feats):
        """Select a 4D feature map tensor for trust head."""
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        # Some backbones return tuple of multi-level feats
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        return feats

    @staticmethod
    def _pixel_to_norm_xy(xy: Tensor, width: int, height: int) -> Tensor:
        """Convert pixel coords (x,y) in [0,W)×[0,H) to normalized [-1,1].

        Uses align_corners=False convention.
        """
        x = xy[..., 0]
        y = xy[..., 1]
        x_norm = (x + 0.5) / float(width) * 2.0 - 1.0
        y_norm = (y + 0.5) / float(height) * 2.0 - 1.0
        return torch.stack([x_norm, y_norm], dim=-1)

    @staticmethod
    def _norm_to_pixel_xy(xy_norm: Tensor, width: int, height: int) -> Tensor:
        x = xy_norm[..., 0]
        y = xy_norm[..., 1]
        x_pix = (x + 1.0) * 0.5 * float(width) - 0.5
        y_pix = (y + 1.0) * 0.5 * float(height) - 0.5
        return torch.stack([x_pix, y_pix], dim=-1)

    @staticmethod
    def _sample_rotation_radians(batch_size: int,
                                 max_deg: float,
                                 device: torch.device,
                                 dtype: torch.dtype) -> Tensor:
        if max_deg <= 0:
            return torch.zeros((batch_size,), device=device, dtype=dtype)
        angles = (torch.rand((batch_size,), device=device, dtype=dtype) * 2.0 -
                  1.0) * float(max_deg)
        return angles * torch.pi / 180.0

    @staticmethod
    def _build_theta_out2in(angle_rad: Tensor) -> Tensor:
        """Return theta for affine_grid: output->input mapping.

        Shape: (B, 2, 3)
        """
        cos = torch.cos(angle_rad)
        sin = torch.sin(angle_rad)
        zeros = torch.zeros_like(cos)
        theta = torch.stack([
            torch.stack([cos, -sin, zeros], dim=-1),
            torch.stack([sin, cos, zeros], dim=-1),
        ],
                           dim=1)
        return theta

    @staticmethod
    def _invert_theta_rotation(theta_out2in: Tensor) -> Tensor:
        """Invert pure-rotation affine theta (no translation)."""
        r = theta_out2in[:, :, :2]  # (B,2,2)
        r_inv = r.transpose(1, 2)
        t_inv = torch.zeros((theta_out2in.size(0), 2, 1),
                            device=theta_out2in.device,
                            dtype=theta_out2in.dtype)
        return torch.cat([r_inv, t_inv], dim=-1)

    def _rotate_inputs(self, inputs: Tensor, angle_rad: Tensor) -> tuple[Tensor, Tensor]:
        """Rotate inputs and return (rotated_inputs, theta_in2out)."""
        theta_out2in = self._build_theta_out2in(angle_rad)
        grid = F.affine_grid(theta_out2in,
                             size=inputs.size(),
                             align_corners=False)
        rotated = F.grid_sample(inputs,
                                grid,
                                mode='bilinear',
                                padding_mode='zeros',
                                align_corners=False)
        theta_in2out = self._invert_theta_rotation(theta_out2in)
        return rotated, theta_in2out

    def _compute_rot_consistency(self, inputs: Tensor,
                                 data_samples: SampleList) -> tuple[Tensor, Tensor, Tensor]:
        """Compute rotation consistency loss and per-joint inconsistency.

        Returns:
            loss_consistency (Tensor): scalar
            inconsistency (Tensor): (B, K) per-joint L2 distance in input pixels
            pseudo_trust (Tensor): (B, K) in [0,1]
        """
        cfg = self.consistency_cfg
        max_deg = float(cfg.get('max_deg', 30.0))
        loss_weight = float(cfg.get('loss_weight', 1.0))
        alpha = float(cfg.get('alpha', 10.0))
        detach_pseudo = bool(cfg.get('detach_pseudo', True))

        b, _, h, w = inputs.shape
        angle_rad = self._sample_rotation_radians(
            b, max_deg=max_deg, device=inputs.device, dtype=inputs.dtype)
        inputs_rot, theta_in2out = self._rotate_inputs(inputs, angle_rad)

        feats = self.extract_feat(inputs)
        feats_rot = self.extract_feat(inputs_rot)

        # Decode keypoints in input space using torch ops.
        # NOTE: RTMCCHead.predict/decode returns numpy arrays, which can't be
        # used for backprop / GPU computation here.
        pred_x, pred_y = self.head.forward(feats)
        pred_x_rot, pred_y_rot = self.head.forward(feats_rot)

        split_ratio = float(getattr(self.head, 'simcc_split_ratio', 1.0))

        def _decode_simcc_torch(px: Tensor, py: Tensor) -> tuple[Tensor, Tensor]:
            # px/py: (B,K,Wx/Wy)
            x_locs = torch.argmax(px, dim=-1).to(dtype=inputs.dtype)
            y_locs = torch.argmax(py, dim=-1).to(dtype=inputs.dtype)
            max_x = torch.amax(px, dim=-1)
            max_y = torch.amax(py, dim=-1)
            scores = torch.minimum(max_x, max_y)
            kpts_xy = torch.stack([x_locs, y_locs], dim=-1) / split_ratio
            return kpts_xy, scores

        kpts, _ = _decode_simcc_torch(pred_x, pred_y)  # (B,K,2)
        kpts_rot, _ = _decode_simcc_torch(pred_x_rot, pred_y_rot)  # (B,K,2)

        # Transform original keypoints into rotated-view coordinates
        kpts_norm = self._pixel_to_norm_xy(kpts, width=w, height=h)  # (B,K,2)
        r = theta_in2out[:, :, :2]  # (B,2,2)
        kpts_rot_from_orig_norm = torch.einsum('bij,bkj->bki', r, kpts_norm)
        kpts_rot_from_orig = self._norm_to_pixel_xy(kpts_rot_from_orig_norm, width=w, height=h)

        diff = kpts_rot - kpts_rot_from_orig
        inconsistency = torch.sqrt((diff**2).sum(dim=-1) + 1e-6)  # (B,K)

        # optional visibility masking from GT
        if cfg.get('use_gt_visible', True):
            vis = []
            for ds in data_samples:
                if hasattr(ds, 'gt_instances') and 'keypoints_visible' in ds.gt_instances:
                    kv = ds.gt_instances.keypoints_visible
                    if isinstance(kv, np.ndarray):
                        kv = torch.from_numpy(kv)
                    vis.append(kv[0].to(device=inconsistency.device,
                                     dtype=inconsistency.dtype))
                else:
                    vis.append(torch.ones((inconsistency.size(1),), device=inconsistency.device))
            vis = torch.stack(vis, dim=0).to(inconsistency.device)
            denom = vis.sum().clamp_min(1.0)
            loss_consistency = (inconsistency * vis).sum() / denom
        else:
            loss_consistency = inconsistency.mean()

        loss_consistency = loss_consistency * loss_weight

        pseudo_trust = torch.exp(-alpha * inconsistency)
        if detach_pseudo:
            pseudo_trust = pseudo_trust.detach()

        return loss_consistency, inconsistency.detach(), pseudo_trust

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        losses = super().loss(inputs, data_samples)

        consistency_enabled = bool(self.consistency_cfg.get('enable', False))
        trust_enabled = bool(self.with_trust and self.trust_cfg.get('enable', False))

        loss_cons = None
        inconsistency = None
        pseudo_trust = None

        if consistency_enabled or trust_enabled:
            # If trust is enabled, we also need pseudo_trust, which is derived
            # from the same rotation inconsistency signal.
            if consistency_enabled or trust_enabled:
                loss_cons, inconsistency, pseudo_trust = self._compute_rot_consistency(
                    inputs, data_samples)

        # Rotation equivariance consistency (2D rotation)
        if consistency_enabled and loss_cons is not None:
            losses['loss_consistency_rot'] = loss_cons
            losses['rot_inconsistency_mean'] = inconsistency.mean()
            losses['pseudo_trust_mean'] = pseudo_trust.mean()

        # Trust head supervised by pseudo trust (from consistency)
        if trust_enabled:
            feats = self.extract_feat(inputs)
            feat = self._as_last_feat(feats)
            trust_pred = self.trust_head(feat)  # (B,K) in [0,1]

            if pseudo_trust is not None:
                if self.trust_cfg.get('detach_pseudo', True):
                    pseudo_trust = pseudo_trust.detach()
                loss_w = float(self.trust_cfg.get('loss_weight', 1.0))
                losses['loss_trust'] = F.binary_cross_entropy(
                    trust_pred, pseudo_trust) * loss_w
                losses['trust_pred_mean'] = trust_pred.mean().detach()

        return losses

    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        # Copy from TopdownPoseEstimator.predict to also attach trust scores
        assert self.with_head, 'The model must have head to perform prediction.'

        if self.test_cfg.get('flip_test', False):
            feats_main = self.extract_feat(inputs)
            feats_flip = self.extract_feat(inputs.flip(-1))
            feats_for_head = [feats_main, feats_flip]
        else:
            feats_main = self.extract_feat(inputs)
            feats_for_head = feats_main

        preds = self.head.predict(feats_for_head, data_samples, test_cfg=self.test_cfg)
        if isinstance(preds, tuple):
            batch_pred_instances, batch_pred_fields = preds
        else:
            batch_pred_instances, batch_pred_fields = preds, None

        results = self.add_pred_to_datasample(batch_pred_instances,
                                              batch_pred_fields, data_samples)

        # Attach per-joint trust scores for evaluation/monitoring
        if self.with_trust:
            feat = self._as_last_feat(feats_main)
            trust_pred = self.trust_head(feat)  # (B,K)
            for i, ds in enumerate(results):
                if hasattr(ds, 'pred_instances'):
                    ds.pred_instances.set_field(trust_pred[i].unsqueeze(0),
                                                'keypoint_trust')

        return results
