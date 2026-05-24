import torch
import torch.nn as nn
import torch.nn.functional as F

class TBAF(nn.Module):
    """
    For each group of 3 channels, treat them as 3 points in batch-space,
    compute pairwise distances, and output those distances as the new features.
    """
    def __init__(self, channels):
        assert channels % 3 == 0
        super().__init__()
        self.groups = channels // 3

    def forward(self, x):
        shape = x.shape
        if x.dim() == 4:
            B, C, H, W = x.shape
            N = B * H * W
            x = x.permute(0, 2, 3, 1).reshape(N, C)   # [N, C]
        elif x.dim() == 2:
            N, C = x.shape
        else:
            raise ValueError(f"Expected 2D or 4D input, got {x.dim()}D")

        # Reshape to groups of 3: [N, groups, 3]
        xg = x.reshape(N, self.groups, 3)
        # treat as three points in N-dim space: [groups, 3, N]
        pts = xg.permute(1, 0, 2).permute(0, 2, 1)  # [groups, 3, N]

        dist01 = torch.norm(pts[:, 0] - pts[:, 1], dim=-1, keepdim=True)
        dist02 = torch.norm(pts[:, 0] - pts[:, 2], dim=-1, keepdim=True)
        dist12 = torch.norm(pts[:, 1] - pts[:, 2], dim=-1, keepdim=True)
        dists = torch.cat([dist01, dist02, dist12], dim=-1)  # [groups, 3]
        out = dists.unsqueeze(0).expand(N, -1, -1).reshape(N, -1)

        if shape != out.shape:
            # restore original shape if it was 4D
            if len(shape) == 4:
                out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return out
