import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from typing import Dict, List, Tuple, Union, Optional, Any
from configuration import CFG


def get_relative_position() -> Tensor:
    pass


class SpanCollator(nn.Module):
    """ Custom Collator for Span Boundary Objective Task
    Args:
        cfg: configuration.CFG
    """
    def __init__(self, cfg: CFG) -> None:
        super(SpanCollator, self).__init__()
        self.cfg = cfg

    def forward(self, inputs: Dict) -> Dict:
        pass


class SBOHead(nn.Module):
    """ Custom Head for Span Boundary Objective Task
    Args:
        cfg: configuration.CFG
    """
    def __init__(self, cfg: CFG) -> None:
        super(SBOHead, self).__init__()
        self.cfg = cfg
        self.head = nn.Sequential(
            nn.Linear(self.cfg.dim_model, self.cfg.dim_ffn, bias=False),
            nn.GELU(),
            nn.LayerNorm(self.cfg.dim_ffn),
            nn.Linear(self.cfg.dim_ffn, 1, bias=False),
        )


    def forward(self, hidden_states: Tensor) -> Tensor:
        pass