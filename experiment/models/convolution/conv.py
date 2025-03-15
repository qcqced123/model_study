"""py module of implementing the convolution for replacing self-attention in transformer (quadratic time complexity)
"""
import torch.nn as nn
from torch import Tensor


class DepthWiseSeparableConv(nn.Module):
    """ depth-wise convolution module for sequence modeling (1D)

    depth-wise convolution: convolution for reducing the time complexity,
                            only weighted sum with each filter and dimension

    point-wise convolution: 1x1 bottle-neck convolution for capturing the relation
                            between each hidden state dimension
    Args:
        channels (int): size of model's latent vector space (~= size of hidden state of model)
        kernel_size (int): window size of convolution filter
        dropout (float): probs of dropout after convolution
    """
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super(DepthWiseSeparableConv, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.depthwise = nn.Conv1d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=self.channels
        )
        self.pointwise = nn.Conv1d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=1
        )
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x: Tensor) -> Tensor:
        # x.shape must be (batch, seq, dim_model)
        x = x.transpose(1, 2)  # convert the dimension to (batch, dim_model, seq)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        return x
