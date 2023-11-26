import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List


class LSTMCell(nn.Module):
    """
    Model class for LSTM(Long-Short Term Memory), Single LSTM Cell
    Add Pre-LayerNorm, Dropout to improve performance
    Math:
        f_t = sigmoid(X_t * W_xf + h_t-1 * W_hf + b_f)
        g_t = tanh(X_t * W_xg + h_t-1 * W_hg + b_g)
        i_t = sigmoid(X_t * W_xi + h_t-1 * W_hi + b_i)
        o_t = sigmoid(X_t * W_xo + h_t-1 * W_ho + b_o)
        c_t = matmul(f_t, c_t-1) + matmul(g_t, i_t)
        h_t = matmul(tanh(c_t), o_t)
    Args:
        input_size: input dimension of each timestamp inputs
        hidden_size: hidden dimension of each timestamp hidden states
        output_dim: output dimension of each timestamp outputs
        dropout: dropout rate, default 0.1s
    """
    def __init__(self, input_size: int, hidden_size: int, output_dim: int, dropout: float = 0.1) -> None:
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim

        # hidden states Weight Matrix, no bias for hidden state
        self.W_hf = nn.Linear(hidden_size, hidden_size, bias=False)  # forgot gate
        self.W_hg = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_hi = nn.Linear(hidden_size, hidden_size, bias=False)  # input gate
        self.W_ho = nn.Linear(hidden_size, hidden_size, bias=False)  # output gate

        # inputs Weight Matrix, bias for inputs
        self.W_xf = nn.Linear(input_size, hidden_size)
        self.W_xg = nn.Linear(input_size, hidden_size)
        self.W_xi = nn.Linear(input_size, hidden_size)
        self.W_xo = nn.Linear(input_size, hidden_size)

        # output Weight Matrix, bias for output
        self.W_y = nn.Linear(hidden_size, output_dim)

        self.Layernorm_f = nn.LayerNorm(hidden_size)
        self.Layernorm_g = nn.LayerNorm(hidden_size)
        self.Layernorm_i = nn.LayerNorm(hidden_size)
        self.Layernorm_o = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs: Tensor, c_states: Tensor, h_states: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass for LSTM cell with additional Layernorm, Dropout
        In LSTM, short term memory h_t is the output of LSTM cell same as y_t in RNN
        But, we return y_t for convenience, which is logit in time t, pass h_t to Fully-Connected layer for logit y_t
        Args:
            inputs: input tensor
            c_states: long term memory state
            h_states: short term memory state
        """
        f_t = torch.sigmoid(self.Layernorm_f(self.W_hf(h_states) + self.W_xf(inputs)))
        g_t = torch.tanh(self.Layernorm_g(self.W_hg(h_states) + self.W_xg(inputs)))
        i_t = torch.sigmoid(self.Layernorm_i(self.W_hi(h_states) + self.W_xi(inputs)))
        o_t = torch.sigmoid(self.Layernorm_o(self.W_ho(h_states) + self.W_xo(inputs)))

        c_t = torch.matmul(f_t, c_states) + torch.matmul(g_t, i_t)
        h_t = torch.matmul(torch.tanh(c_t), o_t)
        y_t = self.W_y(h_t)
        return c_t, h_t, y_t


