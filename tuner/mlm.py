import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Tuple, Union
from experiment.configuration import CFG


class MLMHead(nn.Module):
    """
    Custom Masked Language Model Head for MLM Task, which is used for pre-training Auto-Encoding Model (AE)
    For Encoder, Such as BERT, RoBERTa, ELECTRA, DeBERTa, ...
    Args:
        cfg: configuration.CFG
    Notes:
        class var named "decoder" means denoise masking token for prediction, not Transformer Decoder
    References:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L681
    """
    def __init__(self, cfg: CFG) -> None:
        super(MLMHead, self).__init__()
        self.fc = nn.Linear(cfg.dim_model, cfg.dim_model)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(cfg.dim_model, eps=cfg.layer_norm_eps)
        self.decoder = nn.Linear(cfg.dim_model, cfg.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(cfg.vocab_size))  # for matching vocab size
        self.decoder.bias = self.bias

    def forward(self, hidden_states: Tensor) -> Tensor:
        x = self.fc(hidden_states)
        x = self.gelu(x)
        ln_x = self.layer_norm(x)
        logit = self.decoder(ln_x)
        return logit



