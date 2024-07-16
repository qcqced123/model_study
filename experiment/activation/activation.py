import torch.nn as nn
from torch import Tensor


class ReGLU(nn.Module):
    """pytorch module for ReGLU (ReLU-Gated Linear Unit),

    Args:
        dim_model (int): the dimension of the model's hidden states
        dim_ffn (int): the dimension of the feed-forward network

    Math:
        ReGLU(x) = proj_model(ReLU(proj_relu(x)) * proj_glu(x))

    References:
        https://arxiv.org/pdf/2002.05202.pdf
        https://pytorch.org/docs/stable/generated/torch.nn.GLU.html
        https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
    """
    def __init__(self, dim_model: int, dim_ffn: int = 3072) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.dim_glu = 2 * dim_ffn // 3
        self.proj_glu = nn.Linear(dim_model, self.dim_glu)
        self.proj_relu = nn.Linear(dim_model, self.dim_glu)
        self.proj_model = nn.Linear(self.dim_glu, dim_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj_model(self.relu(self.proj_relu(x)) * self.proj_glu(x))


class SwiGLU(nn.Module):
    """pytorch module for SwiGLU (Swish-Gated Linear Unit),

    Args:
        dim_model (int): the dimension of the model's hidden states
        dim_ffn (int): the dimension of the feed-forward network

    Math:
        SwiGLU(x) = proj_model(Swish(proj_swish(x)) * proj_glu(x))

    References:
        https://arxiv.org/pdf/2002.05202.pdf
        https://pytorch.org/docs/stable/generated/torch.nn.GLU.html
        https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
    """
    def __init__(self, dim_model: int, dim_ffn: int = 3072) -> None:
        super().__init__()
        self.swish = nn.SiLU()
        self.dim_glu = 2 * dim_ffn // 3
        self.proj_glu = nn.Linear(dim_model, self.dim_glu)
        self.proj_swish = nn.Linear(dim_model, self.dim_glu)
        self.proj_model = nn.Linear(self.dim_glu, dim_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj_model(self.swish(self.proj_swish(x)) * self.proj_glu(x))


class GEGLU(nn.Module):
    """pytorch module for GEGLU (GELU-Gated Linear Unit),

    Args:
        dim_model (int): the dimension of the model's hidden states
        dim_ffn (int): the dimension of the feed-forward network

    Math:
        GEGLU(x) = proj_model(GELU(proj_gelu(x)) * proj_glu(x))

    References:
        https://arxiv.org/pdf/2002.05202.pdf
        https://pytorch.org/docs/stable/generated/torch.nn.GLU.html
        https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
    """
    def __init__(self, dim_model: int, dim_ffn: int = 3072) -> None:
        super().__init__()
        self.gelu = nn.GELU()
        self.dim_glu = 2 * dim_ffn // 3
        self.proj_glu = nn.Linear(dim_model, self.dim_glu)
        self.proj_gelu = nn.Linear(dim_model, self.dim_glu)
        self.proj_model = nn.Linear(self.dim_glu, dim_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj_model(self.gelu(self.proj_gelu(x)) * self.proj_glu(x))
