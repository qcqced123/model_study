import torch.nn as nn
from torch import Tensor


class SimCLRProjector(nn.Module):
    """ module of SimCLR non-linear projector layer

    this layer project the last hidden state vector from last layer of encoder into contrastive latent space
    contrastive latent space will be set to be smaller than backbone model's latent space dimension

    Args:
        dim_model (int): dimension of backbone model latent space
        dim_clr (int): dimension of contrastive embedding latent space

    Maths:
        z_i = g(h_i) = w_2•σ(w_1•h_i)
    """
    def __init__(self, dim_model: int, dim_clr: int) -> None:
        super(SimCLRProjector, self).__init__()
        self.dim_model = dim_model
        self.dim_clr = dim_clr
        self.activation_func = nn.ReLU()
        self.projector = nn.Linear(self.dim_model, self.dim_clr, bias=False)
        self.mapper = nn.Linear(self.dim_clr, self.dim_clr, bias=False)

    def forward(self, h: Tensor) -> Tensor:
        activation = self.activation_func(
            self.projector(h)
        )
        embedding = self.mapper(activation)
        return embedding