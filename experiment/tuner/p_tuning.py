import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PromptEncoder(nn.Module):
    """
    This class is implemented P-tuning for boost LLM's NLU performance
    Args:
        inputs: input prompt, which is a sequence of tokens
        masking: masking for select pseudo token, which is param subjected to optimization problem
    Notes:
        nn.Dropout(p=dropout): because this module will be used for instead of nn.Embedding Layer,
                               in common sense, we do not apply Dropout to Inputs Embedding Layer
    References:
        https://arxiv.org/abs/2103.10385
    """
    def __init__(self, inputs: Tensor, masking: Tensor, dim_model: int = 1024, dim_mlp: int = 4096, dropout: float = 0.1) -> None:
        super(PromptEncoder, self).__init__()
        self.inputs = inputs  # for entire input prompt
        self.masking = masking  # for masking select pseudo token
        self.prompt_encoder = nn.Sequential(
            nn.LSTM(
                dim_model,
                dim_model // 2,
                num_layers=1,
                bidirectional=True
            ),
            nn.Linear(dim_model, dim_mlp),
            nn.GELU(),
            nn.Linear(dim_mlp, dim_model),
            nn.GELU(),
        )

    def forward(self) -> Tensor:
        inputs = torch.masked_select(self.inputs, self.masking)
        embeddings = self.prompt_encoder(inputs)
        return embeddings


