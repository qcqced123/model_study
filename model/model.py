import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple

from configuration import CFG
from .model_utils import freeze
from model.abstract_task import AbstractTask
from experiment.tuner.clm import CLMHead
from experiment.tuner.mlm import MLMHead
from experiment.tuner.sbo import SBOHead
from experiment.models.attention.electra import ELECTRA
from experiment.models.attention.spanbert import SpanBERT
from experiment.models.attention.distilbert import DistilBERT


class MaskedLanguageModel(nn.Module, AbstractTask):
    """ Custom Model for MLM Task, which is used for pre-training Auto-Encoding Model (AE)
    You can use backbone model as BERT, DeBERTa, Linear Transformer, Roformer ...

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


class CasualLanguageModel(nn.Module, AbstractTask):
    """ Custom Model for CLM Task, which is used for pre-training Auto-Regressive Model (AR),
    like as GPT, T5 ...

    Notes:
        L = L_CLM (pure language modeling)

    Args:
        cfg: configuration.CFG

    References:
        https://huggingface.co/docs/transformers/main/tasks/language_modeling
        https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L748
    """
    def __init__(self, cfg: CFG) -> None:
        super(CasualLanguageModel, self).__init__()
        self.cfg = cfg
        self.model = self.select_model(cfg.num_layers)
        self.lm_head = CLMHead(cfg)

        self._init_weights(self.model)
        self._init_weights(self.lm_head)

        if self.cfg.load_pretrained:
            self.model.load_state_dict(
                torch.load(cfg.checkpoint_dir + cfg.state_dict),
                strict=True
            )
        if self.cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()

    def feature(self, inputs: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.model(inputs, attention_mask)
        return outputs

    def forward(self, inputs: Tensor, attention_mask: Tensor) -> List[Tensor]:
        last_hidden_states, _ = self.feature(inputs, attention_mask)
        logit = self.lm_head(last_hidden_states)
        return logit


class SpanBoundaryObjective(nn.Module, AbstractTask):
    """ Custom Model for SBO Task, which is used for pre-training Auto-Encoding Model such as SpanBERT
    Original SpanBERT has two tasks, MLM & SBO, so we need to create instance of MLMHead & SBOHead
    You can use backbone model as any encoder attention model, alternative to using MLM

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
    You can use backbone model as any encoder attention model, alternative to using MLM

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
            mask_labels: Tensor = None,
            attention_mask: Tensor = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """ forward pass for generator model
        """
        g_logit, d_inputs, d_labels = self.model.generator_fw(
            inputs,
            labels,
            padding_mask,
            mask_labels,
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
        1) distillation loss, calculated by soft targets & soft predictions
            (nn.KLDIVLoss(reduction='batchmean'))
        2) student loss, calculated by hard targets & hard predictions
            (nn.CrossEntropyLoss(reduction='mean')), same as pure MLM Loss
        3) cosine similarity loss, calculated by student & teacher logit similarity
            (nn.CosineEmbeddingLoss(reduction='mean')), similar as contrastive loss

    References:
        https://arxiv.org/pdf/1910.01108.pdf
        https://github.com/huggingface/transformers/blob/main/examples/research_projects/distillation/distiller.py
    """
    def __init__(self, cfg: CFG) -> None:
        super(DistillationKnowledge, self).__init__()
        self.cfg = CFG
        self.model = DistilBERT(
            self.cfg,
            self.select_model
        )
        self._init_weights(self.model)
        if self.cfg.teacher_load_pretrained:  # for teacher model
            self.model.teacher.load_state_dict(
                torch.load(cfg.checkpoint_dir + cfg.teacher_state_dict),
                strict=False
            )
        if self.cfg.student_load_pretrained:  # for student model
            self.model.student.load_state_dict(
                torch.load(cfg.checkpoint_dir + cfg.student_state_dict),
                strict=True
            )
        if self.cfg.freeze:
            freeze(self.model.teacher)
            freeze(self.model.mlm_head)

        if self.cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()

    def teacher_fw(
        self,
        inputs: Tensor,
        padding_mask: Tensor,
        mask: Tensor,
        attention_mask: Tensor = None,
        is_valid: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """ teacher forward pass to make soft target, last_hidden_state for distillation loss """
        # 1) make soft target
        temperature = 1.0 if is_valid else self.cfg.temperature
        last_hidden_state, t_logit = self.model.teacher_fw(
            inputs,
            padding_mask,
            attention_mask
        )
        last_hidden_state = torch.masked_select(last_hidden_state, ~mask)  # for inverse select
        last_hidden_state = last_hidden_state.view(-1, self.cfg.dim_model)  # flatten last_hidden_state
        soft_target = F.softmax(
            t_logit.view(-1, self.cfg.vocab_size) / temperature**2,  # flatten softmax distribution
            dim=-1
        )  # [bs* seq, vocab_size]
        return last_hidden_state, soft_target

    def student_fw(
        self,
        inputs: Tensor,
        padding_mask: Tensor,
        mask: Tensor,
        attention_mask: Tensor = None,
        is_valid: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """ student forward pass to make soft prediction, hard prediction for student loss """
        temperature = 1.0 if is_valid else self.cfg.temperature
        last_hidden_state, s_logit = self.model.teacher_fw(
            inputs,
            padding_mask,
            attention_mask
        )
        last_hidden_state = torch.masked_select(last_hidden_state, ~mask)  # for inverse select
        last_hidden_state = last_hidden_state.view(-1, self.cfg.dim_model)  # flatten last_hidden_state
        c_labels = last_hidden_state.new(last_hidden_state.size(0)).fill_(1)
        soft_pred = F.softmax(
            s_logit.view(-1, self.cfg.vocab_size) / temperature**2,  # flatten softmax distribution
            dim=-1
        )
        return last_hidden_state, s_logit, soft_pred, c_labels

