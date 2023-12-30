import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from typing import Dict, List, Tuple, Union, Optional, Any
from configuration import CFG


def post_processing(x: Tensor) -> Tensor:
    """ Post Processing for Replaced Token Detection Task """
    pass


class RTDCollator(nn.Module):
    """ Replaced Token Detection Collator (RTD) for Pretraining
    from ELECTRA original paper

    """


class RTDHead(nn.Module):
    """ Replaced Token Detection Head (RTD) for Pretraining from ELECTRA original paper
    RTD Task is same as Binary Classification Task (BCE in pytorch)
    Args:
        cfg: configuration.CFG
    """
    def __init__(self, cfg: CFG) -> None:
        super(RTDHead, self).__init__()
        self.cfg = cfg
        self.classifier = nn.Linear(self.cfg.dim_model, 2)

    def forward(self, hidden_states: Tensor) -> Tensor:
        logit = self.classifier(hidden_states)
        return logit
