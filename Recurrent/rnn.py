import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List


class RecurrentCell(nn.Module):
    """
    Model class for pure Recurrent Neural Network, Single Recurrent Cell
    Add Post-LayerNorm, Dropout to improve performance
    Math:
        h_t = tanh(X_t * W_x + h_t-1 * W_h + b)
        y_t = h_t * W_y + b_y (logit in time t)
    Args:
        input_size: input dimension of each timestamp inputs
        hidden_size: hidden dimension of each timestamp hidden states
        output_dim: output dimension of each timestamp outputs
        dropout: dropout rate, default 0.1s
    """
    def __init__(self, input_size: int, hidden_size: int, output_dim: int, dropout: float = 0.1) -> None:
        super(RecurrentCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.W_x = nn.Linear(input_size, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)  # no bias for hidden state
        self.W_y = nn.Linear(hidden_size, output_dim)
        self.Layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs: Tensor, hidden_states: Tensor) -> Tuple[Tensor, Tensor]:
        """ Forward pass for RNN cell with additional Layernorm, Dropout """
        z_t = self.Layernorm(self.W_h(hidden_states) + self.W_x(inputs))
        h_t = self.dropout(torch.tanh(z_t))
        y_t = self.W_y(h_t)
        return h_t, y_t


