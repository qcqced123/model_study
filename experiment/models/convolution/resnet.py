import torch
import torch.nn as nn
import torch.nn.functional as F
import configuration as CFG

from typing import Tuple
from torch import Tensor
from einops.layers.torch import Rearrange
from experiment.models.abstract_model import AbstractModel


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self):
        return


class ResNetEncoderLayer(nn.Module):
    """
    """
    def __init__(self):
        super(ResNetEncoderLayer, self).__init__()
        self.activation = nn.ReLU()
        self.convolution = None
        self.pooling = MeanPooling()

    def forward(self):
        return


class ResNetEncoder(nn.Module):
    """
    """
    def __init__(
        self,
        cfg: CFG,
        num_layers: int = 12,
        num_features: int = 768,
        batch_norm_eps: float = 0.02,
        hidden_dropout_prob: float = 0.1,
        gradient_checkpointing: bool = False
    ):
        super(ResNetEncoder, self).__init__()
        self.cfg = cfg
        self.num_layers = num_layers
        self.num_features = num_features
        self.batch_norm = nn.BatchNorm2d(
            num_features=num_features,
            eps=batch_norm_eps
        )
        self.hidden_dropout = nn.Dropout(p=hidden_dropout_prob)  # dropout is not learnable
        self.gradient_checkpointing = gradient_checkpointing
        self.layer = nn.ModuleList([
            ResNetEncoderLayer() for _ in range(self.num_layers)
        ])

    def forward(self, inputs: Tensor,) -> Tuple[Tensor, Tensor]:
        x = inputs
        layer_output = []
        for layer in self.layer:
            if self.gradient_checkpointing and self.cfg.train:
                x = self._gradient_checkpointing_func(
                    layer.__call__,
                    x,
                )
            else:
                x = layer(
                    x,
                )
            layer_output.append(x)
        last_hidden_state = self.layer_norm(x)  # because of applying pre-layer norm
        hidden_states = torch.stack(layer_output, dim=0).to(x.device)  # shape: [num_layers, BS, SEQ_LEN, DIM_Model]
        return last_hidden_state, hidden_states


class ResNet(nn.Module):
    """
    """
    def __init__(self, cfg, num_layers: int):
        super(ResNet, self).__init__()
        self.cfg = cfg
        self.num_layers = num_layers

        self.encoder = ResNetEncoder(
            self.cfg,
            self.num_layers,

        )

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """ For cls pooling """
        assert inputs.ndim != 4, f"Input shape should be [BS, CHANNEL, IMAGE_SIZE, IMAGE_SIZE], but got {inputs.shape}"

        # get latent vector of input image
        last_hidden_state, hidden_states = self.encoder(
            inputs
        )
        return last_hidden_state, hidden_states
