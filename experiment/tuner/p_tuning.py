import torch
import torch.nn as nn

from configuration import CFG
from torch import Tensor


class PromptEncoder(nn.Module):
    """
    This class is implemented P-tuning for boost LLM's NLU performance and speed in fine-tuning
    You must freeze original word embedding

    We expected input prompt sentence's shape as below:
        [cls] prompt(pseudo token) [src] context(anchor) [src] prompt(pseudo token) [src] context(anchor) [src] target [src]

    Args:
        c_src: tensor index of context span's start position, same as end index of left template part
        c_end: tensor index of context span's end position, same as start index of right template part

    Notes:
        1) nn.Dropout(p=dropout): Because this module will be used for instead of nn.Embedding Layer,
                                  in common sense, we do not apply Dropout to Inputs Embedding Layer.
                                  So, we do not apply dropout into mlp layer
        2) nn.GELU(): In original paper, author use RELU() but we use GELU because of common sense

    Maths:
        1) h_i = MLP([LSTM(h0:i):LSTM(hi:m)])
        => nn.Linear([Bi-LSTM(h[0:i]):Bi-LSTM(h[i+1:])])
          - two layers of LSTM Module
          - concatenate each other
          - mixture by passing them into mlp layer

    References:
        https://arxiv.org/abs/2103.10385
        https://huggingface.co/docs/peft/package_reference/p_tuning
        https://github.com/huggingface/peft/blob/main/src/peft/tuners/p_tuning/model.py
    """
    def __init__(self, cfg: CFG, c_src: int, c_end: int) -> None:
        super(PromptEncoder, self).__init__()
        self.cfg = cfg
        self.context_start = c_src
        self.context_end = c_end
        self.dim_mlp = self.cfg.virtual_token_dim*2

        self.prompt_embedding = nn.Embedding(self.cfg.num_virtual_tokens, self.cfg.virtual_token_dim)
        self.prompt_encoder = nn.LSTM(
            self.cfg.virtual_token_dim,
            self.cfg.virtual_token_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=self.cfg.hidden_dropout_prob,
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.dim_mlp, self.dim_mlp),
            nn.GELU(),
            nn.Linear(self.dim_mlp, self.cfg.virtual_token_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.prompt_embedding(inputs)
        lh, rh = x[:, :self.context_start, :], x[:, self.context_end+1:, :]
        ph = torch.hstack([self.prompt_encoder(lh), self.prompt_encoder(rh)])
        h = self.mlp(ph)
        return h


