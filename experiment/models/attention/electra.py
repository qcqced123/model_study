import torch
import torch.nn as nn
import torch.nn.functional as F
from experiment.models.abstract_model import AbstractModel
from torch import Tensor
from typing import Tuple, List
from einops.layers.torch import Rearrange
from configuration import CFG


class Generator(nn.Module):
    pass


class Discriminator(nn.Module):
    pass


class ELECTRA(nn.Module, AbstractModel):
    """ Main class for ELECTRA, having all of sub-blocks & modules such as Generator & Discriminator
    Init Scale of ELECTRA Hyper-Parameters, Embedding Layer, Encoder Blocks of Generator, Discriminator
    Var:
        cfg: configuration.CFG
        generator: Generator, which is used for generating replaced tokens for RTD
                   should select backbone model ex) BERT, RoBERTa, DeBERTa, ...
        discriminator: Discriminator, which is used for detecting replaced tokens for RTD
                       should select backbone model ex) BERT, RoBERTa, DeBERTa, ...

    References:
        https://arxiv.org/pdf/2003.10555.pdf
        https://github.com/google-research/electra
    """
    def __init__(self):
        super(ELECTRA, self).__init__()
        self.cfg = CFG
        self.generator = Generator(self.cfg.generator)
        self.discriminator = Discriminator(self.cfg.discrminator)
