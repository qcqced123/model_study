import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from typing import Dict, List, Tuple, Union, Optional, Any
from ..tuner.mlm import PretrainingMaskingCollator
from configuration import CFG


def get_relative_position() -> Tensor:
    pass


class SpanCollator(PretrainingMaskingCollator):
    """ Custom Collator for Span Boundary Objective Task
    Algorithm:
    1) Select 2 random tokens from input tokens for spanning
    2) Calculate relative position embedding for each token with 2 random tokens froms step 1.
    3) Calculate span boundary objective with 2 random tokens from step 1 & pos embedding from step 2.
    Args:
        cfg: configuration.CFG
        span_probability: probability of span length for Geometric Distribution
                         (default: 0.2 => Recommended by original paper)
        max_span_length: maximum span length of each span in one batch sequence
                         (default: 10 => Recommended by original paper)
    References:
        https://arxiv.org/pdf/1907.10529.pdf
    """
    def __init__(self, cfg: CFG, span_probability: float = 0.2, max_span_length: int = 10) -> None:
        super(SpanCollator, self).__init__()
        self.cfg = cfg
        self.tokenizer = self.cfg.tokenizer
        self.span_probability = span_probability
        self.max_span_length = max_span_length

    def get_mask_tokens(
        self,
        inputs: Tensor,
        mask_labels: Tensor
    ) -> Tuple[Any, Any]:
        raise NotImplementedError

    def forward(self, batched: List[Dict[str, Tensor]]) -> Dict:
        """ Abstract Method for Collator, you must implement this method in child class """
        input_ids = [torch.tensor(x["input_ids"]) for x in batched]
        token_type_ids = [torch.tensor(x["token_type_ids"]) for x in batched]
        attention_mask = [torch.tensor(x["attention_mask"]) for x in batched]

        padding_mask = [self.get_padding_mask(x) for x in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

        mask_labels = []
        for x in batched:
            ref_tokens = []
            for input_id in x["input_ids"]:
                token = self.tokenizer._convert_id_to_token(input_id)
                ref_tokens.append(token)
            mask_labels.append(self._whole_word_mask(ref_tokens))

        mask_labels = [torch.tensor(x) for x in mask_labels]
        mask_labels = pad_sequence(mask_labels, batch_first=True, padding_value=0)
        input_ids, labels = self.get_mask_tokens(
            input_ids,
            mask_labels
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "padding_mask": padding_mask,
        }


class SBOHead(nn.Module):
    """ Custom Head for Span Boundary Objective Task, this module return logit value for each token
    Args:
        cfg: configuration.CFG
    References:
        https://arxiv.org/pdf/1907.10529.pdf
    """
    def __init__(self, cfg: CFG) -> None:
        super(SBOHead, self).__init__()
        self.cfg = cfg
        self.head = nn.Sequential(
            nn.Linear(self.cfg.dim_model, self.cfg.dim_ffn, bias=False),
            nn.GELU(),
            nn.LayerNorm(self.cfg.dim_ffn),
            nn.Linear(self.cfg.dim_ffn, self.cfg.dim_model, bias=False),
            nn.GELU(),
            nn.LayerNorm(self.cfg.dim_model),
        )
        self.classifier = nn.Linear(self.cfg.dim_model, self.cfg.vocab_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """ you must pass hidden_states, which is already concatenated with x_s-1, x_e+1, p_i-s+1 """
        assert hidden_states.size(-1) == torch.tensor(self.cfg.dim_model*3), f'Expected last dim size: dim_model*3, but got {hidden_states.size(-1)}'
        z = self.head(hidden_states)
        logit = self.classifier(z)
        return logit
