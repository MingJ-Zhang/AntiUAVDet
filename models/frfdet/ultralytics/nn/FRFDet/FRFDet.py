import torch
import torch.nn as nn
import typing as t
from einops import rearrange

from ultralytics.nn.modules import Conv
from ultralytics.utils.act import build_act

__all__ = ['IBS_D', 'IBS_U', 'SFRCFMul', 'SFRCFAdd']


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    For channel shuffling
    """
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # transpose
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class IBS_D(nn.Module):

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            kernel_size: int = 3,
            stride: int = 2,
            padding: int = 0,
            hidden_ratio: int = 2,
            act: str = 'silu',
            gate: str = 'sigmoid',
            need_shuffle: bool = False
    ):
        super(IBS_D, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.kernel_size = kernel_size
        self.stride = stride
        self.act = build_act(act)()
        self.padding = padding
        self.need_shuffle = need_shuffle
        self.hidden_chans = out_chans * hidden_ratio

        self.conv = nn.Sequential(
            Conv(c1=in_chans, c2=self.hidden_chans, k=1, s=1, g=1, act=self.act),
            Conv(c1=self.hidden_chans, c2=self.hidden_chans, k=kernel_size, s=1, g=self.hidden_chans, act=self.act),
            Conv(c1=self.hidden_chans, c2=out_chans // (stride ** 2), k=1, s=1, act=self.act)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding > 0:
            x = torch.nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding))

        x1 = self.conv[0](x)
        x2 = x1 + self.conv[1](x1)
        out = self.conv[2](x2)
        # (B, C_C, H, W) -> (B, C_C, new_h, new_w, k_h, k_w)
        out = out.unfold(2, self.stride, self.stride).unfold(3, self.stride, self.stride)
        # (B, C_C, new_h, new_w, k_h, k_w) -> (B, C, new_h, new_w)
        out = rearrange(out, 'b c new_h new_w k_h k_w -> b (k_h k_w c) new_h new_w')

        if self.need_shuffle:
            out = channel_shuffle(out, self.stride ** 2)

        return out


class IBS_U(nn.Module):

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            scale_factor: int,
            kernel_size: int = 3,
            hidden_ratio: int = 2,
            act: str = 'silu',
    ):
        super(IBS_U, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.scale_factor = scale_factor
        self.hidden_chans = out_chans * hidden_ratio
        self.act = build_act(act)()

        self.conv = nn.Sequential(
            Conv(c1=in_chans // (self.scale_factor ** 2), c2=self.hidden_chans, k=1, s=1, act=self.act),
            Conv(c1=self.hidden_chans, c2=self.hidden_chans, k=kernel_size, s=1, act=self.act, g=self.hidden_chans),
            Conv(c1=self.hidden_chans, c2=self.out_chans, k=1, s=1, act=self.act)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = rearrange(x, 'b (k_h k_w c) new_h new_w -> b c (new_h k_h) (new_w k_w)',
                        k_h=self.scale_factor, k_w=self.scale_factor)
        out = self.conv(out)
        # y1 = self.conv[0](out)
        # y2 = y1 + self.conv[1](y1)
        # y2 = self.conv[2](y2)
        return out

class SFRCFMul(nn.Module):

    def __init__(
            self,
            in_chans: t.List[int],
            kernel_pair: t.List[int],
            hidden_ratio: int = 2,
            act: str = 'silu',
    ):
        super(SFRCFMul, self).__init__()
        self.pre_chans = [c // 2 for c in in_chans]
        self.post_chans = [c // 2 for c in in_chans]

        self.group_conv1 = nn.ModuleList([
            Conv(c1=self.pre_chans[0], c2=self.pre_chans[0], k=kernel_pair[0], g=self.pre_chans[0]),
            Conv(c1=self.pre_chans[1], c2=self.pre_chans[1], k=kernel_pair[0], g=self.pre_chans[1])
        ])
        self.group_conv2 = nn.ModuleList([
            Conv(c1=self.pre_chans[0], c2=self.pre_chans[0], k=kernel_pair[1], g=self.pre_chans[0]),
            Conv(c1=self.pre_chans[1], c2=self.pre_chans[1], k=kernel_pair[1], g=self.pre_chans[1]),
        ])

        self.point_conv1 = nn.ModuleList([
            nn.Conv2d(self.pre_chans[0], self.post_chans[0], kernel_size=1, bias=True),
            nn.Conv2d(self.pre_chans[1], self.post_chans[1], kernel_size=1, bias=True),
        ])

        self.point_conv2 = nn.ModuleList([
            nn.Conv2d(self.pre_chans[0], self.post_chans[1], kernel_size=1, bias=True),
            nn.Conv2d(self.pre_chans[1], self.post_chans[0], kernel_size=1, bias=True),
        ])

        self.fuse_conv = nn.ModuleList([
            Conv(c1=self.pre_chans[0], c2=in_chans[0], k=1),
            Conv(c1=self.pre_chans[1], c2=in_chans[1], k=1),
        ])

        self.act = build_act(act)()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x1, x2 = inputs[0], inputs[1]

        x1_0, x1_1 = torch.chunk(x1, 2, dim=1)
        x2_0, x2_1 = torch.chunk(x2, 2, dim=1)

        y1_0 = self.group_conv1[0](x1_0)
        y1_0 = self.point_conv1[0](y1_0)
        y1_1 = self.group_conv2[0](x1_1)
        y1_1 = self.point_conv2[0](y1_1)

        y2_0 = self.group_conv1[1](x2_0)
        y2_0 = self.point_conv1[1](y2_0)
        y2_1 = self.group_conv2[1](x2_1)
        y2_1 = self.point_conv2[1](y2_1)

        y1 = y1_0 * self.act(y2_1)
        y1 = x1 + self.fuse_conv[0](y1)
        y2 = y2_0 * self.act(y1_1)
        y2 = x2 + self.fuse_conv[1](y2)

        return torch.cat((y1, y2), dim=1)


class SFRCFAdd(nn.Module):

    def __init__(
            self,
            in_chans: t.List[int],
            kernel_pair: t.List[int],
            hidden_ratio: int = 2,
            act: str = 'silu',
    ):
        super(SFRCFAdd, self).__init__()
        self.pre_chans = [c // 2 for c in in_chans]
        self.post_chans = [c // 2 for c in in_chans]

        self.group_conv1 = nn.ModuleList([
            Conv(c1=self.pre_chans[0], c2=self.pre_chans[0], k=kernel_pair[0], g=self.pre_chans[0]),
            Conv(c1=self.pre_chans[1], c2=self.pre_chans[1], k=kernel_pair[0], g=self.pre_chans[1])
        ])
        self.group_conv2 = nn.ModuleList([
            Conv(c1=self.pre_chans[0], c2=self.pre_chans[0], k=kernel_pair[1], g=self.pre_chans[0]),
            Conv(c1=self.pre_chans[1], c2=self.pre_chans[1], k=kernel_pair[1], g=self.pre_chans[1]),
        ])

        self.point_conv1 = nn.ModuleList([
            nn.Conv2d(self.pre_chans[0], self.post_chans[0], kernel_size=1, bias=True),
            nn.Conv2d(self.pre_chans[1], self.post_chans[1], kernel_size=1, bias=True),
        ])

        self.point_conv2 = nn.ModuleList([
            nn.Conv2d(self.pre_chans[0], self.post_chans[1], kernel_size=1, bias=True),
            nn.Conv2d(self.pre_chans[1], self.post_chans[0], kernel_size=1, bias=True),
        ])

        self.fuse_conv = nn.ModuleList([
            Conv(c1=self.pre_chans[0], c2=in_chans[0], k=1),
            Conv(c1=self.pre_chans[1], c2=in_chans[1], k=1),
        ])

        self.act = build_act(act)()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x1, x2 = inputs[0], inputs[1]

        x1_0, x1_1 = torch.chunk(x1, 2, dim=1)
        x2_0, x2_1 = torch.chunk(x2, 2, dim=1)

        y1_0 = self.group_conv1[0](x1_0)
        y1_0 = self.point_conv1[0](y1_0)
        y1_1 = self.group_conv2[0](x1_1)
        y1_1 = self.point_conv2[0](y1_1)

        y2_0 = self.group_conv1[1](x2_0)
        y2_0 = self.point_conv1[1](y2_0)
        y2_1 = self.group_conv2[1](x2_1)
        y2_1 = self.point_conv2[1](y2_1)

        y1 = y1_0 + self.act(y2_1)
        y1 = x1 + self.fuse_conv[0](y1)
        y2 = y2_0 + self.act(y1_1)
        y2 = x2 + self.fuse_conv[1](y2)

        return torch.cat((y1, y2), dim=1)
