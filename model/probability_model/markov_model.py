import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List, Dict, Optional, Any


class MarkovModel(nn.Module):
    """ Markov Model for time series forecasting
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super(MarkovModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.W_x = nn.Linear(input_dim, hidden_dim)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_y = nn.Linear(hidden_dim, output_dim)
        self.Layernorm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs: Tensor, hidden_states: Tensor) -> Tuple[Tensor, Tensor]:
        z_t = self.Layernorm(self.W_h(hidden_states) + self.W_x(inputs))
        h_t = self.dropout(torch.tanh(z_t))
        y_t = self.W_y(h_t)
        return h_t, y_t
