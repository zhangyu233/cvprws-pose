import torch
import torch.nn as nn

from mmpose.registry import MODELS


@MODELS.register_module()
class TrustHead(nn.Module):
    """Minimal joint-level trust head (placeholder).

    Input: a feature map (N, C, H, W)
    Output: per-joint trust scores (N, K) in [0, 1]
    """

    def __init__(self, in_channels: int, num_keypoints: int = 17):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, num_keypoints),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.pool(feat).flatten(1)
        return self.fc(x)
