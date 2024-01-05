import torch
import torch.nn as nn
import torch.nn.functional as F
from experiment.models.abstract_model import AbstractModel
from torch import Tensor
from typing import Tuple, List
from einops.layers.torch import Rearrange
from ..attention import deberta
from configuration import CFG


class SpanBERTEncoder(nn.Module):
    """ SpanBERT Encoder, In original paper, BERT is used as backbone model but we select DeBERTa as backbone model
    you can change backbone model to any other model easily, just passing other model name to cfg.encoder_name
    But, you must pass ONLY encoder model such as BERT, RoBERTa, DeBERTa, ...
    Args:
        cfg: configuration.CFG
    """
    def __init__(self, cfg: CFG) -> None:
        super(SpanBERTEncoder, self).__init__()
        self.cfg = cfg
        self.model_name = cfg.span_encoder_name
        self.enc_model = getattr(deberta, self.model_name)(self.cfg)

    def forward(self, inputs: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        assert inputs.ndim == 2, f'Expected (batch, sequence) got {inputs.shape}'
        word_embeddings, rel_pos_emb, abs_pos_emb = self.enc_model.embeddings(
            inputs
        )
        last_hidden_state, hidden_states = self.enc_model.encoder(
            word_embeddings,
            rel_pos_emb,
            padding_mask,
            attention_mask
        )
        emd_hidden_states = hidden_states[-self.cfg.num_emd]
        emd_last_hidden_state, emd_hidden_states = self.enc_model.emd_encoder(
            emd_hidden_states,
            abs_pos_emb,
            rel_pos_emb,
            padding_mask,
            attention_mask
        )
        return emd_last_hidden_state, emd_hidden_states


class SpanBERT(nn.Module, AbstractModel):
    """ Main class for SpanBERT, having all of sub-blocks & modules such as self-attention, feed-forward, BERTEncoder ..
    Init Scale of SpanBERT Hyper-Parameters, Embedding Layer, Encoder Blocks

    In original paper, BERT is used as backbone model but we select DeBERTa as backbone model
    you can change backbone model to any other model easily, just passing other model name to cfg.encoder_name
    But, you must pass ONLY encoder model such as BERT, RoBERTa, DeBERTa, ...

    Args:
        cfg: configuration.CFG
        backbone: baseline architecture (nn.Module) which is used for Pretraining SpanBERT
    References:
        https://arxiv.org/pdf/1907.10529.pdf
    """
    def __init__(self, cfg: CFG, backbone: nn.Module) -> None:
        super(SpanBERT, self).__init__()
        self.cfg = cfg
        self.encoder = SpanBERTEncoder(self.cfg)
        self.gradient_checkpointing = self.cfg.gradient_checkpoint

    def forward(self, inputs: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        emd_last_hidden_state, emd_hidden_states = self.encoder(
            inputs,
            padding_mask,
            attention_mask
        )
        return emd_last_hidden_state, emd_hidden_states

