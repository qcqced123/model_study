import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from typing import Dict, List, Tuple, Union, Optional, Any
from configuration import CFG


class PreTrainingCollator(nn.Module):
    """ Custom Collator for Pretraining
    Masked Language Modeling Algorithm with Dynamic Masking from RoBERTa
        1) 15% of input tokens are selected at random for prediction
        2) 80% of the selected tokens are replaced with [MASK] token
        3) 10% of the selected tokens are replaced with random token in vocabulary
        4) The remaining 10% are left unchanged
    Args:
        cfg: configuration.CFG
    """
    def __init__(self, cfg: CFG) -> None:
        super(PreTrainingCollator, self).__init__()
        self.tokenizer = cfg.tokenizer

    @staticmethod
    def get_padding_mask(input_id: Tensor) -> Tensor:
        return torch.zeros(input_id.shape).bool()

    def get_mask_tokens(
        self,
        input_ids: Tensor,
        mlm_probability: float = 0.15,
        special_tokens_mask: Optional[Any] = None
    ) -> Tuple[Tensor, Tensor]:
        """ Get Masked Tokens for MLM Task
        """
        input_ids = input_ids.clone()
        labels = input_ids.clone()

        # 1) 15% of input tokens are selected at random for prediction
        probability_matrix = torch.full(labels.shape, mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        # 2) 80% of the selected tokens are replaced with [MASK] token
        indices_replaced = (
                torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = (
                torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
                & masked_indices
                & ~indices_replaced
        )
        # 3) 10% of the selected tokens are replaced with random token in vocabulary
        random_words = torch.randint(
            self.tokenizer.vocab_size, labels.shape, dtype=torch.long
        )
        input_ids[indices_random] = random_words[indices_random]

        return input_ids, labels


class MLMCollator(PreTrainingCollator):
    """ Custom Collator for MLM Task, which is used for pre-training Auto-Encoding Model (AE)
    Dataloader returns a list to collator, so collator should be able to handle list of tensors
    Args:
        cfg: configuration.CFG
        special_tokens_mask: special tokens mask for masking
    References:
        https://huggingface.co/docs/transformers/v4.32.0/ko/tasks/masked_language_modeling
        https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L607
        https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L748
    """
    def __init__(self, cfg: CFG, special_tokens_mask: Optional[Any] = None):
        super(MLMCollator, self).__init__(cfg)
        self.cfg = cfg
        self.special_tokens_mask = special_tokens_mask

    def __call__(self, batched: List[Dict[str, Tensor]]):
        input_ids = [torch.tensor(x["input_ids"]) for x in batched]
        token_type_ids = [torch.tensor(x["token_type_ids"]) for x in batched]
        attention_mask = [torch.tensor(x["attention_mask"]) for x in batched]

        padding_mask = [self.get_padding_mask(x) for x in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        input_ids, labels = self.get_mask_tokens(
            input_ids, self.cfg.mlm_probability, special_tokens_mask=self.special_tokens_mask
        )
        padding_mask = pad_sequence(padding_mask, batch_first=True, padding_value=True)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "padding_mask": padding_mask,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask
        }


class MLMHead(nn.Module):
    """
    Custom Masked Language Model Head for MLM Task, which is used for pre-training Auto-Encoding Model (AE)
    For Encoder, Such as BERT, RoBERTa, ELECTRA, DeBERTa, ... etc
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



