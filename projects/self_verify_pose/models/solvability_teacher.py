from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class SolvabilityOutputs:
    # Per-joint reprojection error after fitting: (B, K)
    reproj_err: Tensor
    # Scalar objective per sample: (B,)
    energy: Tensor
    # Pseudo trust in [0,1]: (B, K)
    pseudo_trust: Tensor


class SolvabilityTeacher:
    """A lightweight IK-style teacher for a solvability trust signal.

    This is a *minimal* implementation to make the idea runnable:
    we fit 3D joint positions with weak-perspective projection under
    bone-length and simple joint-angle constraints.

    It is designed for training-time pseudo labels (detach by default).
    """

    # COCO-17 indices:
    # 5 L_shoulder, 6 R_shoulder, 7 L_elbow, 8 R_elbow, 9 L_wrist, 10 R_wrist
    # 11 L_hip, 12 R_hip, 13 L_knee, 14 R_knee, 15 L_ankle, 16 R_ankle

    def __init__(
        self,
        num_iters: int = 15,
        lr: float = 0.05,
        w_reproj: float = 1.0,
        w_bone: float = 10.0,
        w_angle: float = 1.0,
        w_reg: float = 1e-3,
        alpha: float = 10.0,
        eps: float = 1e-6,
        detach: bool = True,
        use_gt_visible: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.num_iters = int(num_iters)
        self.lr = float(lr)
        self.w_reproj = float(w_reproj)
        self.w_bone = float(w_bone)
        self.w_angle = float(w_angle)
        self.w_reg = float(w_reg)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.detach = bool(detach)
        self.use_gt_visible = bool(use_gt_visible)
        self.device = device

        # A small constraint graph with cycles makes solvability non-trivial.
        # Edges are (i, j) pairs.
        self.edges = torch.tensor(
            [
                [5, 7], [7, 9],
                [6, 8], [8, 10],
                [11, 13], [13, 15],
                [12, 14], [14, 16],
                [5, 6],
                [11, 12],
                [5, 11],
                [6, 12],
            ],
            dtype=torch.long,
        )
        # Rough template bone-length ratios (arbitrary units). A per-sample
        # scale variable will adapt globally.
        self.template_L = torch.tensor(
            [
                0.30, 0.25,
                0.30, 0.25,
                0.42, 0.42,
                0.42, 0.42,
                0.35,
                0.30,
                0.50,
                0.50,
            ],
            dtype=torch.float32,
        )

        # Angle constraints at elbows/knees: (parent, joint, child)
        self.angle_triplets = torch.tensor(
            [
                [5, 7, 9],
                [6, 8, 10],
                [11, 13, 15],
                [12, 14, 16],
            ],
            dtype=torch.long,
        )
        self.min_angle_deg = 5.0
        self.max_angle_deg = 175.0

    @staticmethod
    def _safe_norm(x: Tensor, eps: float) -> Tensor:
        return torch.sqrt((x * x).sum(dim=-1) + eps)

    def _angle_penalty(self, p: Tensor) -> Tensor:
        """Soft penalty for joint angles outside [min, max].

        Args:
            p: (B,K,3)
        Returns:
            penalty: (B,)
        """
        tri = self.angle_triplets.to(device=p.device)
        pa, pj, pc = tri[:, 0], tri[:, 1], tri[:, 2]

        a = p[:, pa] - p[:, pj]  # (B,T,3)
        b = p[:, pc] - p[:, pj]
        an = a / self._safe_norm(a, self.eps).unsqueeze(-1)
        bn = b / self._safe_norm(b, self.eps).unsqueeze(-1)
        cos = (an * bn).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)  # (B,T)
        ang = torch.arccos(cos) * 180.0 / torch.pi

        min_a = self.min_angle_deg
        max_a = self.max_angle_deg
        below = F.softplus((min_a - ang) / 5.0)
        above = F.softplus((ang - max_a) / 5.0)
        return (below + above).mean(dim=-1)

    def compute(
        self,
        kpts_2d: Tensor,
        visible: Optional[Tensor] = None,
    ) -> SolvabilityOutputs:
        """Fit a simple 3D explanation and return reprojection residuals.

        Args:
            kpts_2d: (B,K,2) in pixels (or any consistent 2D units).
            visible: optional (B,K) in {0,1}.
        """
        device = self.device or kpts_2d.device
        kpts_2d = kpts_2d.to(device)
        b, k, _ = kpts_2d.shape

        vis = None
        if visible is not None:
            vis = visible.to(device=device, dtype=kpts_2d.dtype)
        else:
            vis = torch.ones((b, k), device=device, dtype=kpts_2d.dtype)

        # Initialize: center 2D and set z=0
        mean_2d = kpts_2d.mean(dim=1, keepdim=True)
        p_xy0 = kpts_2d - mean_2d
        p0 = torch.zeros((b, k, 3), device=device, dtype=kpts_2d.dtype)
        p0[..., :2] = p_xy0

        # Weak-perspective parameters
        s0 = torch.ones((b, 1), device=device, dtype=kpts_2d.dtype)
        t0 = mean_2d.squeeze(1)

        # Per-sample skeleton scale
        bone_scale0 = torch.ones((b, 1), device=device, dtype=kpts_2d.dtype)

        p = p0.clone().detach().requires_grad_(True)
        s = s0.clone().detach().requires_grad_(True)
        t = t0.clone().detach().requires_grad_(True)
        bone_scale = bone_scale0.clone().detach().requires_grad_(True)

        edges = self.edges.to(device=device)
        L = self.template_L.to(device=device, dtype=kpts_2d.dtype)

        opt = torch.optim.Adam([p, s, t, bone_scale], lr=self.lr)

        for _ in range(self.num_iters):
            proj = s.unsqueeze(-1) * p[..., :2] + t.unsqueeze(1)
            reproj = ((proj - kpts_2d) ** 2).sum(dim=-1)  # (B,K)
            loss_reproj = (reproj * vis).sum() / (vis.sum().clamp_min(1.0))

            # Bone lengths
            pi = p[:, edges[:, 0]]
            pj = p[:, edges[:, 1]]
            bone_len = self._safe_norm(pi - pj, self.eps)  # (B,E)
            target = bone_scale * L.unsqueeze(0)  # (B,E)
            loss_bone = ((bone_len - target) ** 2).mean()

            loss_angle = self._angle_penalty(p).mean()

            loss_reg = (p * p).mean() + ((bone_scale - 1.0) ** 2).mean()

            loss = (
                self.w_reproj * loss_reproj
                + self.w_bone * loss_bone
                + self.w_angle * loss_angle
                + self.w_reg * loss_reg
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            # Keep scale positive
            with torch.no_grad():
                s.clamp_(min=1e-3)
                bone_scale.clamp_(min=1e-3)

        with torch.no_grad():
            proj = s.unsqueeze(-1) * p[..., :2] + t.unsqueeze(1)
            err = torch.sqrt(((proj - kpts_2d) ** 2).sum(dim=-1) + self.eps)  # (B,K)

            # scalar energy (masked mean reprojection error + penalties)
            energy = (err * vis).sum(dim=-1) / (vis.sum(dim=-1).clamp_min(1.0))

            pseudo_trust = torch.exp(-self.alpha * err)

        if self.detach:
            err = err.detach()
            energy = energy.detach()
            pseudo_trust = pseudo_trust.detach()

        return SolvabilityOutputs(reproj_err=err, energy=energy, pseudo_trust=pseudo_trust)
