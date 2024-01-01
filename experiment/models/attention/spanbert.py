import torch
import torch.nn as nn
import torch.nn.functional as F
from experiment.models.abstract_model import AbstractModel
from torch import Tensor
from typing import Tuple, List
from einops.layers.torch import Rearrange
from configuration import CFG


class SpanBERT(nn.Module, AbstractModel):
    """ Main class for SpanBERT, having all of sub-blocks & modules such as self-attention, feed-forward, BERTEncoder ..
    Init Scale of SpanBERT Hyper-Parameters, Embedding Layer, Encoder Blocks
    Args:
    References:
    """
    def __init__(self, cfg: CFG) -> None:
        super(SpanBERT, self).__init__()
        self.cfg = cfg

    def forward(self, inputs: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> Tensor:
        pass

