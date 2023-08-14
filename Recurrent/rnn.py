import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RecurrentModel(nn.Module):
    """
    Model class for pure Recurrent Neural Network
    Variable:
        X_t: input at time t
        W_x: weight for input X, same value in one unique RNN cell
        W_h: weight for hidden state, same value in one unique RNN cell
        W_y: weight for output, same value in one unique RNN cell
        b: bias for hidden state, same value in one unique RNN cell
        b_y: bias for output, same value in one unique RNN cell
    Math:
        h_t = tanh(X_t * W_x + h_t-1 * W_h + b)
        y_t = h_t * W_y + b_y (logit in time t)
        p_t = softmax(y_t) (probability in time t)
        L_t = -log(p_t) (loss in time t)
    Args:

    """
    def __init__(self, input_size: int, hidden_size: int, output_dim: int) -> None:
        super(RecurrentModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.x_weight = nn.Linear(self.input_size, self.hidden_size)
        self.h_weight = F.tanh(nn.Linear(self.hidden_size, self.hidden_size) + self.x_weight)
        self.y_weight = nn.Linear(self.hidden_size, self.output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        First Cell must be initialized with zero hidden state
        """
        count = 0
        if count == 0:
            emb_x = self.x_weight(inputs)
            init_emb_h = self.h_weight(emb_x)
            output = self.y_weight(init_emb_h)

