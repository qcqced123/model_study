import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from typing import Dict, List, Tuple, Union, Optional, Any

from .mlm import WholeWordMaskingCollator
from configuration import CFG


class CLMCollator(WholeWordMaskingCollator):

    def __init__(self, cfg: CFG) -> None:
        super(CLMCollator, self).__init__(cfg=cfg)

    def get_mask_tokens(self, inputs: Tensor, pad_mask: Tensor) -> Tensor:
        """ make masking matrix for Decoder Masked Multi-Head Self-attention
        combine padding mask and attention mask, naming as attention_mask
        """
        lm_mask = torch.tril(torch.ones(inputs.shape[0], inputs.shape[-1], inputs.shape[-1]))
        attention_mask = pad_mask * lm_mask
        return attention_mask

    def forward(self, batched: List[Dict[str, Tensor]]) -> Dict:
        """ masking for CLM Task
        return:
            input_ids: padded input_ids
            labels: rolled input_ids
            attention_mask: padded attention_mask
        """
        input_ids = [x["input_ids"] for x in batched]
        padding_mask = [self.get_padding_mask(x) for x in input_ids]

        padding_mask = pad_sequence(padding_mask, batch_first=True, padding_value=True)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        labels = input_ids.clone().roll(-1, dims=-1)
        attention_mask = self.get_mask_tokens(input_ids, padding_mask)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class CLMHead(nn.Module):
    """
    Custom Casual Language Model Head for CLM Task, which is used for pre-training Auto-Regressive Model (AR)
    For Decoder, Such as GPT2, GPTNeo, ...
    Args:
        cfg: configuration.CFG
    References:
        https://huggingface.co/docs/transformers/main/tasks/language_modeling.html
    """

    def __init__(self, cfg: CFG) -> None:
        super(CLMHead, self).__init__()
        self.cfg = cfg
        self.decoder = nn.Linear(cfg.dim_model, cfg.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(cfg.vocab_size))  # for matching vocab size
        self.decoder.bias = self.bias

    def forward(self, hidden_states: Tensor) -> Tensor:
        x = hidden_states
        logits = self.decoder(x)
        return logits


