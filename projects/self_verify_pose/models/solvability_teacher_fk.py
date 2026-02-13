from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class SolvabilityFKOutputs:
    reproj_err: Tensor  # (B, K)
    energy: Tensor  # (B,)
    pseudo_trust: Tensor  # (B, K)


class SolvabilityFKTeacher:
    """IK-style teacher based on an explicit kinematic chain (FK via pytorch_kinematics).

    Notes:
    - This is a minimal, runnable FK-based teacher.
    - It fits joint angles + weak-perspective camera (s,t) + global scale.
    - The kinematic model is a lightweight COCO-17-ish skeleton (not SMPL).

    If `pytorch_kinematics` is unavailable, this class raises an ImportError
    with a clear installation hint.
    """

    def __init__(
        self,
        num_iters: int = 20,
        lr: float = 0.05,
        w_reproj: float = 1.0,
        w_limits: float = 0.1,
        w_reg: float = 1e-4,
        alpha: float = 10.0,
        eps: float = 1e-6,
        detach: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.num_iters = int(num_iters)
        self.lr = float(lr)
        self.w_reproj = float(w_reproj)
        self.w_limits = float(w_limits)
        self.w_reg = float(w_reg)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.detach = bool(detach)
        self.device = device

        try:
            import pytorch_kinematics as pk  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                'pytorch_kinematics is required for SolvabilityFKTeacher. '\
                'Install it via: pip install -U pytorch-kinematics\n'\
                f'Original import error: {e}'
            )

        # A tiny URDF capturing a COCO-like skeleton. Units are arbitrary.
        # We use fixed links for facial keypoints relative to head.
        urdf = _COCO17_TINY_URDF
        self._chain = pk.build_chain_from_urdf(urdf)
        self._chain_device: Optional[torch.device] = None
        if self.device is not None:
          self._ensure_chain_device(self.device)

        # Order of joint parameters in the chain
        self._joint_names: List[str] = list(self._chain.get_joint_parameter_names())
        self._dof = len(self._joint_names)

        # We will read positions of these link frames as joint locations.
        # COCO-17 order: [nose, leye, reye, lear, rear, lsho, rsho, lelb, relb,
        #                 lwri, rwri, lhip, rhip, lknee, rknee, lank, rank]
        self._link_names_coco17 = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
        ]

        # Joint limits in radians in the same order as self._joint_names.
        # These are heuristic and mainly to keep IK stable.
        lims = []
        for name in self._joint_names:
            # hinge elbows/knees
            if 'elbow' in name or 'knee' in name:
                lims.append((0.0, 2.8))
            # shoulders/hips allow wider range
            elif 'shoulder' in name or 'hip' in name:
                lims.append((-2.2, 2.2))
            # neck
            elif 'neck' in name:
                lims.append((-1.2, 1.2))
            else:
                lims.append((-3.14, 3.14))
        self._limits = torch.tensor(lims, dtype=torch.float32)  # (DOF,2)

    def _ensure_chain_device(self, device: torch.device) -> None:
        """Move the internal kinematic chain to `device` if needed."""
        if self._chain_device == device:
            return
        if hasattr(self._chain, 'to'):
          # pytorch_kinematics.Chain.to interprets a single positional arg as
          # dtype, so we must pass device as a keyword.
          self._chain = self._chain.to(device=device)
        self._chain_device = device

    def _limits_penalty(self, theta: Tensor) -> Tensor:
        # theta: (B, DOF)
        lim = self._limits.to(device=theta.device, dtype=theta.dtype)
        min_v = lim[:, 0].unsqueeze(0)
        max_v = lim[:, 1].unsqueeze(0)
        below = F.softplus((min_v - theta) / 0.2)
        above = F.softplus((theta - max_v) / 0.2)
        return (below + above).mean(dim=-1)  # (B,)

    def compute(self, kpts_2d: Tensor, visible: Optional[Tensor] = None) -> SolvabilityFKOutputs:
        device = self.device or kpts_2d.device
        self._ensure_chain_device(torch.device(device))
        kpts_2d = kpts_2d.to(device=device, dtype=torch.float32)
        b, k, _ = kpts_2d.shape
        if k != 17:
            raise ValueError(f'SolvabilityFKTeacher expects COCO-17 keypoints, got K={k}')

        if visible is None:
            vis = torch.ones((b, k), device=device, dtype=torch.float32)
        else:
            vis = visible.to(device=device, dtype=torch.float32)

        # init camera from 2D mean
        mean_2d = kpts_2d.mean(dim=1)
        s = torch.ones((b, 1), device=device, dtype=torch.float32, requires_grad=True)
        t = mean_2d.clone().detach().requires_grad_(True)  # (B,2)
        scale = torch.ones((b, 1), device=device, dtype=torch.float32, requires_grad=True)

        theta = torch.zeros((b, self._dof), device=device, dtype=torch.float32, requires_grad=True)

        opt = torch.optim.Adam([theta, s, t, scale], lr=self.lr)

        for _ in range(self.num_iters):
            # FK
            th_dict = {n: theta[:, i] for i, n in enumerate(self._joint_names)}
            fk = self._chain.forward_kinematics(th_dict)

            # Gather COCO-17 joint 3D positions from link frames
            pts3d = []
            for ln in self._link_names_coco17:
                if ln not in fk:
                    raise KeyError(f'FK output missing link: {ln}')
                mat = fk[ln].get_matrix()  # (B,4,4)
                pts3d.append(mat[:, :3, 3])
            p = torch.stack(pts3d, dim=1)  # (B,17,3)

            p = p * scale.unsqueeze(1)
            proj = s.unsqueeze(-1) * p[..., :2] + t.unsqueeze(1)

            sq = ((proj - kpts_2d) ** 2).sum(dim=-1)  # (B,17)
            loss_reproj = (sq * vis).sum() / (vis.sum().clamp_min(1.0))

            loss_lim = self._limits_penalty(theta).mean()
            loss_reg = (theta * theta).mean() + ((scale - 1.0) ** 2).mean()

            loss = self.w_reproj * loss_reproj + self.w_limits * loss_lim + self.w_reg * loss_reg

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            with torch.no_grad():
                s.clamp_(min=1e-3)
                scale.clamp_(min=1e-3)

        with torch.no_grad():
            th_dict = {n: theta[:, i] for i, n in enumerate(self._joint_names)}
            fk = self._chain.forward_kinematics(th_dict)
            pts3d = []
            for ln in self._link_names_coco17:
                mat = fk[ln].get_matrix()
                pts3d.append(mat[:, :3, 3])
            p = torch.stack(pts3d, dim=1) * scale.unsqueeze(1)
            proj = s.unsqueeze(-1) * p[..., :2] + t.unsqueeze(1)
            err = torch.sqrt(((proj - kpts_2d) ** 2).sum(dim=-1) + self.eps)
            energy = (err * vis).sum(dim=-1) / (vis.sum(dim=-1).clamp_min(1.0))
            pseudo = torch.exp(-self.alpha * err)

        if self.detach:
            err = err.detach()
            energy = energy.detach()
            pseudo = pseudo.detach()

        return SolvabilityFKOutputs(reproj_err=err, energy=energy, pseudo_trust=pseudo)


# A tiny COCO-17-ish skeleton URDF.
# Link frames are used as joint locations; facial points are fixed offsets.
_COCO17_TINY_URDF = r"""
<robot name="coco17_tiny">
  <link name="root"/>

  <!-- torso -->
  <joint name="neck_yaw_j" type="revolute">
    <parent link="root"/>
    <child link="neck"/>
    <origin xyz="0 0.5 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.2" upper="1.2" effort="1" velocity="1"/>
  </joint>
  <link name="neck"/>

  <joint name="neck_pitch_j" type="revolute">
    <parent link="neck"/>
    <child link="head"/>
    <origin xyz="0 0.25 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.2" upper="1.2" effort="1" velocity="1"/>
  </joint>
  <link name="head"/>

  <!-- facial keypoints as fixed frames -->
  <joint name="nose_fix_j" type="fixed">
    <parent link="head"/>
    <child link="nose"/>
    <origin xyz="0 0.10 0.10" rpy="0 0 0"/>
  </joint>
  <link name="nose"/>

  <joint name="leye_fix_j" type="fixed">
    <parent link="head"/>
    <child link="left_eye"/>
    <origin xyz="-0.05 0.12 0.08" rpy="0 0 0"/>
  </joint>
  <link name="left_eye"/>

  <joint name="reye_fix_j" type="fixed">
    <parent link="head"/>
    <child link="right_eye"/>
    <origin xyz="0.05 0.12 0.08" rpy="0 0 0"/>
  </joint>
  <link name="right_eye"/>

  <joint name="lear_fix_j" type="fixed">
    <parent link="head"/>
    <child link="left_ear"/>
    <origin xyz="-0.12 0.10 0.00" rpy="0 0 0"/>
  </joint>
  <link name="left_ear"/>

  <joint name="rear_fix_j" type="fixed">
    <parent link="head"/>
    <child link="right_ear"/>
    <origin xyz="0.12 0.10 0.00" rpy="0 0 0"/>
  </joint>
  <link name="right_ear"/>

  <!-- shoulders from neck -->
  <joint name="left_shoulder_pitch_j" type="revolute">
    <parent link="neck"/>
    <child link="left_shoulder"/>
    <origin xyz="-0.20 0.00 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-2.2" upper="2.2" effort="1" velocity="1"/>
  </joint>
  <link name="left_shoulder"/>

  <joint name="left_shoulder_yaw_j" type="revolute">
    <parent link="left_shoulder"/>
    <child link="left_upperarm"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.2" upper="2.2" effort="1" velocity="1"/>
  </joint>
  <link name="left_upperarm"/>

  <joint name="left_elbow_j" type="revolute">
    <parent link="left_upperarm"/>
    <child link="left_elbow"/>
    <origin xyz="-0.30 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="2.8" effort="1" velocity="1"/>
  </joint>
  <link name="left_elbow"/>

  <joint name="left_wrist_fix_j" type="fixed">
    <parent link="left_elbow"/>
    <child link="left_wrist"/>
    <origin xyz="-0.25 0 0" rpy="0 0 0"/>
  </joint>
  <link name="left_wrist"/>

  <joint name="right_shoulder_pitch_j" type="revolute">
    <parent link="neck"/>
    <child link="right_shoulder"/>
    <origin xyz="0.20 0.00 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-2.2" upper="2.2" effort="1" velocity="1"/>
  </joint>
  <link name="right_shoulder"/>

  <joint name="right_shoulder_yaw_j" type="revolute">
    <parent link="right_shoulder"/>
    <child link="right_upperarm"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.2" upper="2.2" effort="1" velocity="1"/>
  </joint>
  <link name="right_upperarm"/>

  <joint name="right_elbow_j" type="revolute">
    <parent link="right_upperarm"/>
    <child link="right_elbow"/>
    <origin xyz="0.30 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="2.8" effort="1" velocity="1"/>
  </joint>
  <link name="right_elbow"/>

  <joint name="right_wrist_fix_j" type="fixed">
    <parent link="right_elbow"/>
    <child link="right_wrist"/>
    <origin xyz="0.25 0 0" rpy="0 0 0"/>
  </joint>
  <link name="right_wrist"/>

  <!-- hips from root -->
  <joint name="left_hip_pitch_j" type="revolute">
    <parent link="root"/>
    <child link="left_hip"/>
    <origin xyz="-0.10 0.0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-2.2" upper="2.2" effort="1" velocity="1"/>
  </joint>
  <link name="left_hip"/>

  <joint name="left_hip_yaw_j" type="revolute">
    <parent link="left_hip"/>
    <child link="left_thigh"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.2" upper="2.2" effort="1" velocity="1"/>
  </joint>
  <link name="left_thigh"/>

  <joint name="left_knee_j" type="revolute">
    <parent link="left_thigh"/>
    <child link="left_knee"/>
    <origin xyz="0 -0.42 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="2.8" effort="1" velocity="1"/>
  </joint>
  <link name="left_knee"/>

  <joint name="left_ankle_fix_j" type="fixed">
    <parent link="left_knee"/>
    <child link="left_ankle"/>
    <origin xyz="0 -0.42 0" rpy="0 0 0"/>
  </joint>
  <link name="left_ankle"/>

  <joint name="right_hip_pitch_j" type="revolute">
    <parent link="root"/>
    <child link="right_hip"/>
    <origin xyz="0.10 0.0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-2.2" upper="2.2" effort="1" velocity="1"/>
  </joint>
  <link name="right_hip"/>

  <joint name="right_hip_yaw_j" type="revolute">
    <parent link="right_hip"/>
    <child link="right_thigh"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.2" upper="2.2" effort="1" velocity="1"/>
  </joint>
  <link name="right_thigh"/>

  <joint name="right_knee_j" type="revolute">
    <parent link="right_thigh"/>
    <child link="right_knee"/>
    <origin xyz="0 -0.42 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="2.8" effort="1" velocity="1"/>
  </joint>
  <link name="right_knee"/>

  <joint name="right_ankle_fix_j" type="fixed">
    <parent link="right_knee"/>
    <child link="right_ankle"/>
    <origin xyz="0 -0.42 0" rpy="0 0 0"/>
  </joint>
  <link name="right_ankle"/>
</robot>
"""
