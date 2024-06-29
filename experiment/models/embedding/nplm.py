import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple, List


class NPLM(nn.Module):
    """ module for the Neural Probabilistic Language Model (NPLM) from Bengio et al. (2003)

    this embedding model is quite simple and is used as a baseline for many NLP tasks,
    very similar to the modern transformer model with causal language modeling

    Architecture:
        1) project the discrete input word tensors to continuously word embedding dimension
            - embedding tensor's hidden state is the noted as m (=dim_embedding in source code)

        2) project the word embedding tensor into model hidden state dimension
        3) concatenate the embedding alongside with hidden state dimension (not yet implemented, but can be added)
        4) pass the embedding tensor through a non-linear activation function (tanh)
        5) pass the embedding tensor to the projector W (=self.fc in source code)
        6) project the embedding tensor to the vocabulary size dimension (=self.decoder in source code)
        7) get softmax probability distribution from 5) + 6) output tensor

    Args:
        cfg: configuration object for the model

    References:

    """
    def __init__(self, cfg) -> None:
        super(NPLM, self).__init__()
        self.cfg = cfg
        self.word_embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.dim_embedding)
        self.proj_h = nn.Linear(self.cfg.dim_embedding, self.cfg.dim_hidden)
        self.decoder = nn.Linear(self.cfg.dim_hidden, self.cfg.vocab_size, bias=False)
        self.fc = nn.Linear(self.cfg.dim_embedding, self.cfg.vocab_size)
        self.activation = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        # check input tensor shape is valid
        assert x.dim() == 2, f"Input tensor must be 2D (batch_size, seq_len), got {x.dim()}D tensor"

        word_emb = self.word_embedding(x)
        hx = self.activation(self.proj_h(word_emb))
        logit = self.decoder(hx) + self.fc(x)
        return logit






