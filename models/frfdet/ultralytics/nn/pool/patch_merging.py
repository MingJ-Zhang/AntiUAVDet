import torch
import torch.nn as nn
from einops import rearrange

__all__ = ['PatchMerging']


class PatchMerging(nn.Module):
    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.reduction = nn.Linear(4 * in_chans, out_chans, bias=False)
        self.norm = norm_layer(4 * in_chans)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        # Dynamic calculation of half size based on H and W
        half_H, half_W = H // 2, W // 2

        x = x.permute(0, 2, 3, 1)

        # Slice the input into four quadrants for patch merging
        x0 = x[:, 0::2, 0::2, :]  # B, H/2, W/2, C
        x1 = x[:, 1::2, 0::2, :]  # B, H/2, W/2, C
        x2 = x[:, 0::2, 1::2, :]  # B, H/2, W/2, C
        x3 = x[:, 1::2, 1::2, :]  # B, H/2, W/2, C

        # Concatenate the 4 quadrants along the last dimension (C dimension)
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # B, H/2, W/2, 4*C

        # Reshape the tensor to (B, H/2*W/2, 4*C)
        x = x.reshape(B, -1, 4 * C)  # B, H/2*W/2, 4*C

        # Apply normalization and reduction (linear transformation)
        x = self.norm(x)
        x = self.reduction(x)
        x = rearrange(x, 'B (H W) C -> B C H W', H=half_H, W=half_W)
        # x = x.view(B, half_H, half_W, 2 * C).permute(0, 3, 1, 2)

        return x
