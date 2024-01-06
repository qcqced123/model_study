import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Dict, Tuple
from experiment.tuner.mlm import MLMHead
from experiment.tuner.sbo import SBOHead
from configuration import CFG
from model.abstract_task import AbstractTask
from experiment.models.attention.electra import ELECTRA
from experiment.models.attention.spanbert import SpanBERT


class MaskedLanguageModel(nn.Module, AbstractTask):
    """ Custom Model for MLM Task, which is used for pre-training Auto-Encoding Model (AE)
    Args:
        cfg: configuration.CFG
    References:
        https://huggingface.co/docs/transformers/main/tasks/masked_language_modeling
        https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L748
    """
    def __init__(self, cfg: CFG) -> None:
        super(MaskedLanguageModel, self).__init__()
        self.cfg = cfg
        self.model = self.select_model()
        self.mlm_head = MLMHead(cfg)
        if self.cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()

        self._init_weights(self.model)
        self._init_weights(self.mlm_head)

        if self.cfg.load_pretrained:
            self.model.load_state_dict(
                torch.load(cfg.checkpoint_dir + cfg.state_dict),
                strict=True
            )

    def feature(self, inputs: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> Tensor:
        outputs = self.model(inputs, padding_mask)
        return outputs

    def forward(self, inputs: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> List[Tensor]:
        last_hidden_states, _ = self.feature(inputs, padding_mask)
        logit = self.mlm_head(last_hidden_states)
        return logit


class ReplacedTokenDetection(nn.Module, AbstractTask):
    """ Custom Model for RTD Task, which is used for pre-training Auto-Encoding Model such as ELECTRA
    Args:
        cfg: configuration.CFG
    """
    def __init__(self, cfg: CFG) -> None:
        super(ReplacedTokenDetection, self).__init__()
        self.cfg = cfg
        self.model = ELECTRA(
            self.cfg,
            self.select_model
        )
        if self.cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()

        self._init_weights(self.model)
        if self.cfg.generator_load_pretrained:  # for generator
            self.model.generator.load_state_dict(
                torch.load(cfg.checkpoint_dir + cfg.state_dict),
                strict=True
            )
        if self.cfg.discriminator_load_pretrained:  # for discriminator
            self.model.discriminator.load_state_dict(
                torch.load(cfg.checkpoint_dir + cfg.state_dict),
                strict=True
            )

    def forward(
            self,
            inputs: Tensor,
            labels: Tensor,
            padding_mask: Tensor,
            attention_mask: Tensor = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        g_logit, d_logit, d_labels = self.model(
            inputs,
            labels,
            padding_mask,
            attention_mask
        )
        return g_logit, d_logit, d_labels


class SpanBoundaryObjective(nn.Module, AbstractTask):
    """ Custom Model for SBO Task, which is used for pre-training Auto-Encoding Model such as SpanBERT
    Original SpanBERT has two tasks, MLM & SBO, so we need to create instance of MLMHead & SBOHead
    Notes:
        L_span = L_MLM + L_SBO
    Args:
        cfg: configuration.CFG
    References:
        https://arxiv.org/pdf/1907.10529.pdf
    """
    def __init__(self, cfg: CFG) -> None:
        super(SpanBoundaryObjective, self).__init__()
        self.cfg = cfg
        self.model = SpanBERT(
            self.cfg,
            self.select_model()
        )
        self.mlm_head = MLMHead(self.cfg)
        self.sbo_head = SBOHead(self.cfg)
        if self.cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()

        self._init_weights(self.model)
        self._init_weights(self.mlm_head)
        self._init_weights(self.sbo_head)

        if self.cfg.load_pretrained:
            self.model.load_state_dict(
                torch.load(cfg.checkpoint_dir + cfg.state_dict),
                strict=True
            )

    def feature(self, inputs: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        outputs = self.model(inputs, padding_mask)
        return outputs

    def forward(
        self,
        inputs: Tensor,
        padding_mask: Tensor,
        mask_labels: Tensor,
        attention_mask: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        last_hidden_states, _ = self.feature(inputs, padding_mask)
        mlm_logit = self.mlm_head(last_hidden_states)
        sbo_logit = self.sbo_head(
            last_hidden_states,
            mask_labels
        )
        return mlm_logit, sbo_logit
