import torch
import torch.nn as nn
import torch.nn.functional as F



class LightAngleRotate(nn.Module):
    """
    Splits channels in half. B becomes a rotation direction,
    a learned bias gate controls the rotation strength (0..1).
    """
    def __init__(self, channels, max_angle=torch.pi/4):
        super().__init__()
        assert channels % 2 == 0, "Channels must be even"
        self.half = channels // 2
        self.max_angle = max_angle
        # Per‑pair learnable bias for the gate (scalar per pair)
        self.gate_bias = nn.Parameter(torch.zeros(self.half))

    def forward(self, x):
        A = x[:, :self.half]        # first half
        B = x[:, self.half:]        # second half

        # Rotation direction from B
        theta = torch.tanh(B) * self.max_angle   # same shape as A/B

        # Gate: per‑pair learned bias
        gate = torch.sigmoid(self.gate_bias)     # shape (half,)

        # Reshape gate to be broadcastable: (1, half, 1, 1) for 4D, (1, half) for 2D
        # Use view to match the channel dimension
        if A.dim() == 4:
            gate = gate.view(1, self.half, 1, 1)
        elif A.dim() == 2:
            gate = gate.view(1, self.half)
        else:
            raise ValueError(f"Expected 2D or 4D input, got {A.dim()}D")

        # Effective rotation angle = gate * direction
        angle = gate * theta

        # Rotate
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        A_rot = A * cos_a - B * sin_a
        B_rot = A * sin_a + B * cos_a

        return torch.cat([A_rot, B_rot], dim=1)
    
class TriangleDistActivation(nn.Module):
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