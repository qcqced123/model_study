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
    Algorithm:
    1) Select 2 random tokens from input tokens for spanning
    2) Calculate relative position embedding for each token with 2 random tokens froms step 1.
    3) Calculate span boundary objective with 2 random tokens from step 1 & pos embedding from step 2.
    Args:
        cfg: configuration.CFG
    """
    def __init__(self, cfg: CFG) -> None:
        super(SpanCollator, self).__init__()
        self.cfg = cfg

    def forward(self, inputs: Dict) -> Dict:
        pass


class SBOHead(nn.Module):
    """ Custom Head for Span Boundary Objective Task, this module return logit value for each token
    Args:
        cfg: configuration.CFG
    References:
        https://arxiv.org/pdf/1907.10529.pdf
    """
    def __init__(self, cfg: CFG) -> None:
        super(SBOHead, self).__init__()
        self.cfg = cfg
        self.head = nn.Sequential(
            nn.Linear(self.cfg.dim_model, self.cfg.dim_ffn, bias=False),
            nn.GELU(),
            nn.LayerNorm(self.cfg.dim_ffn),
            nn.Linear(self.cfg.dim_ffn, self.cfg.dim_model, bias=False),
            nn.GELU(),
            nn.LayerNorm(self.cfg.dim_model),
        )
        self.classifier = nn.Linear(self.cfg.dim_model, self.cfg.vocab_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        z = self.head(hidden_states)
        logit = self.classifier(z)
        return logit
