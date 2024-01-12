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
from experiment.models.attention.distilbert import DistilBERT


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
        self.model = self.select_model(cfg.num_layers)
        self.mlm_head = MLMHead(cfg)

        self._init_weights(self.model)
        self._init_weights(self.mlm_head)

        if self.cfg.load_pretrained:
            self.model.load_state_dict(
                torch.load(cfg.checkpoint_dir + cfg.state_dict),
                strict=True
            )
        if self.cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()

    def feature(self, inputs: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> Tensor:
        outputs = self.model(inputs, padding_mask)
        return outputs

    def forward(self, inputs: Tensor, padding_mask: Tensor, attention_mask: Tensor = None) -> List[Tensor]:
        last_hidden_states, _ = self.feature(inputs, padding_mask)
        logit = self.mlm_head(last_hidden_states)
        return logit


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
            self.select_model(cfg.num_layers)
        )
        self.mlm_head = MLMHead(self.cfg)
        self.sbo_head = SBOHead(self.cfg)

        self._init_weights(self.model)
        self._init_weights(self.mlm_head)
        self._init_weights(self.sbo_head)

        if self.cfg.load_pretrained:
            self.model.load_state_dict(
                torch.load(cfg.checkpoint_dir + cfg.state_dict),
                strict=True
            )
        if self.cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()

    def feature(
        self,
        inputs: Tensor,
        padding_mask: Tensor,
        attention_mask: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """ Extract feature embedding from backbone model
        """
        outputs = self.model(inputs, padding_mask)
        return outputs

    def forward(
        self,
        inputs: Tensor,
        padding_mask: Tensor,
        mask_labels: Tensor,
        attention_mask: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """ Forwarding inputs into model & return 2 types of logit
        """
        last_hidden_states, _ = self.feature(inputs, padding_mask)
        mlm_logit = self.mlm_head(last_hidden_states)
        sbo_logit = self.sbo_head(
            last_hidden_states,
            mask_labels
        )
        return mlm_logit, sbo_logit


class ReplacedTokenDetection(nn.Module, AbstractTask):
    """ Custom Model for RTD Task, which is used for pre-training Auto-Encoding Model such as ELECTRA

    We add 3 task options:
        1) select masking method:
            - pure MLM (Sub-Word Masking)
            - WWM (Whole Word Masking)
            - SBO (Span Boundary Objective)
        2) select backbone model: BERT, DeBERTa, ...
        3) select sharing embedding method:
            - ES (Embedding Sharing)
            - GDES (Generator Discriminator Embedding Sharing)

    you can select any other 3 options in config json file
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
        self._init_weights(self.model)
        if self.cfg.generator_load_pretrained:  # for generator
            self.model.generator.load_state_dict(
                torch.load(cfg.checkpoint_dir + cfg.state_dict),
                strict=False
            )
        if self.cfg.discriminator_load_pretrained:  # for discriminator
            self.model.discriminator.load_state_dict(
                torch.load(cfg.checkpoint_dir + cfg.state_dict),
                strict=True
            )
        if self.cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()

    def generator_fw(
            self,
            inputs: Tensor,
            labels: Tensor,
            padding_mask: Tensor,
            attention_mask: Tensor = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """ forward pass for generator model
        """
        g_logit, d_inputs, d_labels = self.model.generator_fw(
            inputs,
            labels,
            padding_mask,
            attention_mask
        )
        return g_logit, d_inputs, d_labels

    def discriminator_fw(
            self,
            inputs: Tensor,
            padding_mask: Tensor,
            attention_mask: Tensor = None
    ) -> Tensor:
        """ forward pass for discriminator model
        """
        d_logit = self.model.discriminator_fw(
            inputs,
            padding_mask,
            attention_mask
        )
        return d_logit


class DistillationKnowledge(nn.Module, AbstractTask):
    """ Custom Task Module for Knowledge Distillation by DistilBERT Style Architecture
    DistilBERT Style Architecture is Teacher-Student Framework for Knowledge Distillation,

    And then they have 3 objective functions:
        1) distillation loss, calculated bys soft targets & soft predictions
        2) student loss, calculated by hard targets & hard predictions
        3) cosine similarity loss, calculated by student & teacher logit similarity


    """
    def __init__(self, cfg: CFG) -> None:
        super(DistillationKnowledge, self).__init__()
        self.cfg = CFG
        self.model = DistilBERT(
            self.cfg,
            self.select_model
        )
        self._init_weights(self.model)
        if self.cfg.teacher_load_pretrained:  # for generator
            self.model.teacher.load_state_dict(
                torch.load(cfg.checkpoint_dir + cfg.teacher_state_dict),
                strict=False
            )
        if self.cfg.student_load_pretrained:  # for discriminator
            self.model.student.load_state_dict(
                torch.load(cfg.checkpoint_dir + cfg.student_state_dict),
                strict=True
            )
        if self.cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self) -> Tensor:
        pass
