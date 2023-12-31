import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from typing import Dict, List, Tuple, Union, Optional, Any
from configuration import CFG


def get_discriminator_input(inputs: Tensor, labels: Tensor, pred: Tensor) -> Tuple[Tensor, Tensor]:
    """ Post Processing for Replaced Token Detection Task
    1) get index of the highest probability of [MASK] token
    2) convert [MASK] token to prediction token
    3) make label for Discriminator
    Args:
        inputs: pure inputs from tokenizing by tokenizer
        labels: labels for masked language modeling
        pred: prediction tensor from Generator
    returns:
        inputs: Tensor
        label: Tensor
    """
    # 1) flatten pred to 2D Tensor
    d_inputs, d_labels = inputs.clone().view(-1), None
    flat_pred, flat_label = pred.view(-1, pred.size(-1)), labels.view(-1)  # (batch * sequence, vocab_size)

    # 2) get index of the highest probability of [MASK] token
    pred_token_idx, mlm_mask_idx = flat_pred.argmax(dim=-1), torch.where(flat_label != -100)
    mask = flat_label.ge(-100)
    pred_tokens = torch.masked_select(pred_token_idx, mask)  # select [MASK] token, return element

    # 3) convert [MASK] token to prediction token
    d_inputs[mlm_mask_idx] = pred_tokens
    d_inputs = d_inputs.view(-1, pred.size(0))  # covert to [batch, sequence]
    d_labels = torch.eq(inputs, flat_pred).long()  # unnecessary recover to original label shape, keep flatten state
    return d_inputs, d_labels


class RTDCollator(nn.Module):
    """ Replaced Token Detection Collator (RTD) for Pretraining
    from ELECTRA original paper
    """
    pass


class RTDHead(nn.Module):
    """ Replaced Token Detection Head (RTD) for Pretraining from ELECTRA original paper
    RTD Task is same as Binary Classification Task (BCE in pytorch)
    classes: 0 (replaced) or 1 (original)
    Args:
        cfg: configuration.CFG
    """
    def __init__(self, cfg: CFG) -> None:
        super(RTDHead, self).__init__()
        self.cfg = cfg
        self.classifier = nn.Linear(self.cfg.dim_model, 2)

    def forward(self, hidden_states: Tensor) -> Tensor:
        logit = self.classifier(hidden_states)
        return logit
