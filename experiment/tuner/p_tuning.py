import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PromptEncoder(nn.Module):
    """
    This class is implemented P-tuning for boost LLM's NLU performance in fine-tuning

    Notes:
        1) nn.Dropout(p=dropout): because this module will be used for instead of nn.Embedding Layer,
                               in common sense, we do not apply Dropout to Inputs Embedding Layer
        2) nn.GELU(): In original paper, author use RELU() but we use GELU because of common sense

    Maths:
        1) bi-lstm(x[0:t], x[t+1:])

    References:
        https://arxiv.org/abs/2103.10385
    """
    def __init__(self, dim_model: int = 1024, dropout: float = 0.1) -> None:
        super(PromptEncoder, self).__init__()
        self.prompt_encoder = nn.LSTM(
            dim_model,
            dim_model // 2,
            num_layers=2,
            bidirectional=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.GELU(),
            nn.Linear(dim_model, dim_model),
        )

    def forward(self, inputs: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            inputs: input prompt, which is a sequence of tokens
            mask: masking for select pseudo token, which is param subjected to optimization problem
        """
        x = torch.masked_select(inputs, mask)
        x = self.prompt_encoder(x)
        p_h = self.mlp(x)
        return p_h


