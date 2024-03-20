import torch.nn as nn
from torch import Tensor


class SwiGLU(nn.Module):
    """ Pytorch module for SwiGLU (Swish-Gated Linear Unit), Swish(SILU) + GLU

    Args:
        dim_ffn: int, the dimension of the feed-forward network

    Math:
        SwiGLU(x) = SiLU(x) + (1 - GLU(x))*MLP(x)
        SwiGLU(x) = x*sigmoid(beta * x) + (1 - sigmoid(beta*x)) * (Wx + b)

    References:
        https://arxiv.org/pdf/2002.05202.pdf
        https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
        https://pytorch.org/docs/stable/generated/torch.nn.GLU.html
    """
    def __init__(self, dim_ffn: int = 3072):
        super().__init__()
        self.silu = nn.SiLU()
        self.glu = nn.GLU()
        self.fc = nn.Linear(dim_ffn, dim_ffn)

    def forward(self, x: Tensor) -> Tensor:
        return self.silu(x) + (1 - self.glu(x)) * self.fc(x)
