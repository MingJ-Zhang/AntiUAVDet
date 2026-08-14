import math
import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ConditionalPoolingLayer']


class ConditionalPoolingLayer(nn.Module):
    def __init__(self, channels: int, size: int, stride: t.Optional[int] = None, padding: int = 0):
        super(ConditionalPoolingLayer, self).__init__()
        self.size = size
        self.stride = stride if stride is not None else size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = self.size
        S = self.stride

        batch = x.shape[0] 
        chan = x.shape[1]

        W = int((x.shape[2] - N) / S) + 1
        H = int((x.shape[3] - N) / S) + 1

        x_kernels = F.unfold(x, kernel_size=self.size, stride=self.stride, padding=self.padding).reshape(batch, chan,
                                                                                                         N * N, -1)
        x_kernels = x_kernels.permute(0, 1, 3, 2)

        mean = x_kernels.mean(dim=3, keepdim=True)

        # Count the number of values greater and less than the mean
        H_num = torch.sum(torch.greater(x_kernels, mean), dim=3, keepdim=True).to(x.dtype)
        L_num = torch.sum(torch.less(x_kernels, mean), dim=3, keepdim=True).to(x.dtype)

        # Perform Conditional-Pooling based on the number of values greater and less than the mean
        x_pooled = torch.where(torch.greater(H_num, L_num),
                               torch.divide(torch.sum(torch.where(torch.greater(x_kernels, mean), x_kernels, 0), dim=3,
                                                      keepdim=True), H_num),
                               torch.where(torch.less(H_num, L_num),
                                           torch.divide(
                                               torch.sum(torch.where(torch.less(x_kernels, mean), x_kernels, 0), dim=3,
                                                         keepdim=True), L_num), mean))

        # Reshape the output to the desired dimensions
        x_pooled = x_pooled.reshape(batch, chan, W, H)
        return x_pooled