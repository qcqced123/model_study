import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List


class RecurrentCell(nn.Module):
    """
    Model class for pure recurrent Neural Network, Single recurrent Cell
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
        self.Layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs: Tensor, hidden_states: Tensor) -> Tensor:
        """ Forward pass for RNN cell with additional Layernorm, Dropout """
        z_t = self.Layernorm(self.W_h(hidden_states) + self.W_x(inputs))
        h_t = self.dropout(torch.tanh(z_t))
        return h_t


class Recurrent(nn.Module):
    """
    Model class for pure recurrent Neural Network, Stacked recurrent Cells

    Math:
        h_t = tanh(X_t * W_x + h_t-1 * W_h + b)
        y_t = h_t * W_y + b_y (logit in time t)

    Args:
        input_size: input dimension of each timestamp inputs
        hidden_size: hidden dimension of each timestamp hidden states
        output_dim: output dimension of each timestamp outputs
        num_layers: number of recurrent layers
        dropout: dropout rate, default 0.1s
    """
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_dim: int,
            num_layers: int,
            dropout: float = 0.1
    ) -> None:
        super(Recurrent, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers = nn.ModuleList(
            [RecurrentCell(input_size, hidden_size, hidden_size, dropout) for _ in range(num_layers)]
        )  # num_layers same as nums of stacked RNN cells
        self.W_y = nn.Linear(hidden_size, output_dim)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """ Forward pass for RNN with stacked RNN cells
        returns:
            x: last hidden states of stacked RNN cells
            y: logit of last hidden states of stacked RNN cells
        """
        x = inputs
        for layer in self.layers:
            h_t = torch.zeros(x.size(0), self.hidden_size).to(x.device)  # initialize hidden states with zeros for each layer of RNN in time step 0
            intermediate_h = []
            for t in range(inputs.size(1)):
                h_t = layer(x[:, t, :], h_t)  # [batch_size, hidden_size]
                intermediate_h.append(h_t.unsqueeze(1))  # [batch_size, 1, hidden_size]
            x = torch.cat(intermediate_h, dim=1)  # making new inputs next layer of RNN
        y = self.W_y(x)
        return x, y
