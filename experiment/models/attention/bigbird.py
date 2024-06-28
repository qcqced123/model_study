import torch
import torch.nn as nn
import torch.nn.functional as F
from experiment.models.abstract_model import AbstractModel
from torch import Tensor
from typing import Tuple, List
from einops.layers.torch import Rearrange
from configuration import CFG


def random_attention():
    """ function for the random attention in bigbird from google research

    Args:


    References:
        https://arxiv.org/pdf/2007.14062
        https://huggingface.co/blog/big-bird
    """
    pass


def local_attention():
    """ function for the local attention in bigbird from google research

    Args:


    References:
        https://arxiv.org/pdf/2007.14062
        https://huggingface.co/blog/big-bird
    """
    pass


def global_attention():
    """ function for the global attention in bigbird from google research
    """
    pass


def block_sparse_attention():
    """ function for the block sparse attention in bigbird from google research
    """


class BlockSparseAttentionHead(nn.Module):
    def __init__(self):
        super(BlockSparseAttentionHead, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        pass