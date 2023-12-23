import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List, Dict, Optional, Any


class MarkovModel(nn.Module):
    """ Markov Model for time series forecasting
    """
    def __init__(self) -> None:
        super(MarkovModel, self).__init__()

    def forward(self, inputs: Tensor, hidden_states: Tensor) -> None:
        pass
