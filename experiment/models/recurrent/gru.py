import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List


class GRUCell(nn.Module):
    """
    Model class for GRU(Gated recurrent Unit), Single GRU Cell
    Add Pre-LayerNorm, Dropout to improve performance
    Math:
        r_t = sigmoid(X_t * W_xr + h_t-1 * W_hr + b_r)
        z_t = sigmoid(X_t * W_xz + h_t-1 * W_hz + b_z)
        g_t = tanh(X_t * W_xg + r_t * (h_t-1 * W_hg + b_g))
        y_t = (1 - z_t) * g_t + z_t * h_t-1
    Args:
        input_size: input dimension of each timestamp inputs
        hidden_size: hidden dimension of each timestamp hidden states
        output_dim: output dimension of each timestamp outputs
        dropout: dropout rate, default 0.1s
    """
    def __init__(self, input_size: int, hidden_size: int, output_dim: int, dropout: float = 0.1) -> None:
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim

        # hidden states Weight Matrix, no bias for hidden state
        self.W_hr = nn.Linear(hidden_size, hidden_size, bias=False)  # forgot gate
        self.W_hz = nn.Linear(hidden_size, hidden_size, bias=False)  # input gate
        self.W_hg = nn.Linear(hidden_size, hidden_size, bias=False)

        # inputs Weight Matrix, bias for inputs
        self.W_xr = nn.Linear(input_size, hidden_size)
        self.W_xz = nn.Linear(input_size, hidden_size)
        self.W_xg = nn.Linear(input_size, hidden_size)

        self.Layernorm_r = nn.LayerNorm(hidden_size)
        self.Layernorm_z = nn.LayerNorm(hidden_size)
        self.Layernorm_g = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs: Tensor, hidden_states: Tensor) -> Tensor:
        """
        Forward pass for LSTM cell with additional Layernorm, Dropout
        Args:
            inputs: input tensor
            hidden_states: latent hidden states for previous timestamp
        """
        r_t = torch.sigmoid(self.Layernorm_r(self.W_hr(hidden_states) + self.W_xr(inputs)))
        z_t = torch.sigmoid(self.Layernorm_z(self.W_hz(hidden_states) + self.W_xz(inputs)))
        g_t = torch.tanh(self.Layernorm_g(self.W_hg(torch.matmul(r_t, hidden_states)) + self.W_xg(inputs)))

        h_t = torch.matmul((1 - z_t), g_t) + torch.matmul(hidden_states, z_t)
        return h_t


class GRU(nn.Module):
    """
    Model class for GRU(Gated recurrent Unit), Stacked GRU Cells
    Args:
        input_size: input dimension of each timestamp inputs`
        hidden_size: hidden dimension of each timestamp hidden states
        output_dim: output dimension of each timestamp outputs
        num_layers: number of recurrent layers
        dropout: dropout rate, default 0.1s
    Notes:
        num_layers same as nums of stacked GRU cells
    """
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_dim: int,
            num_layers: int,
            dropout: float = 0.1
    ) -> None:
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers = nn.ModuleList(
            [GRUCell(input_size, hidden_size, hidden_size, dropout) for _ in range(num_layers)]
        )  # num_layers same as nums of stacked GRU cells
        self.W_y = nn.Linear(hidden_size, output_dim)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """ Forward pass for GRU with stacked GRU cells """
        x = inputs
        for layer in self.layers:
            h_t = torch.zeros(x.size(0), self.hidden_size).to(
                x.device)  # initialize hidden states with zeros for each layer of RNN in time step 0
            intermediate_h = []
            for t in range(inputs.size(1)):
                h_t = layer(x[:, t, :], h_t)  # [batch_size, hidden_size]
                intermediate_h.append(h_t.unsqueeze(1))  # [batch_size, 1, hidden_size]
            x = torch.cat(intermediate_h, dim=1)  # making new inputs next layer of RNN
        y = self.W_y(x)
        return x, y
