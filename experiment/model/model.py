import torch
import torch.nn as nn
import model.pooling as pooling
from torch import Tensor
from transformers import AutoConfig, AutoModel
from typing import List, Dict
import configuration
from model.model_utils import freeze, reinit_topk


class TextRegressor(nn.Module):
    """
    Model Class for Text Regression Task Pipeline
    This class apply reinit_top_k encoder's weight, init weights of fine-tune stage (fully-connect, regressor-head)ß
    Args:
        cfg: configuration.CFG
    """
    def __init__(self, cfg: configuration.CFG) -> None:
        super().__init__()
        self.cfg = cfg
        self.auto_cfg = AutoConfig.from_pretrained(
            cfg.model,
            output_hidden_states=True
        )
        self.auto_cfg.attention_probs_dropout_prob = self.cfg.attention_probs_dropout_prob
        self.auto_cfg.hidden_dropout_prob = self.cfg.hidden_dropout_prob

        self.model = AutoModel.from_pretrained(
            cfg.model,
            config=self.auto_cfg
        )
        self.model.resize_token_embeddings(len(self.cfg.tokenizer))
        self.pooling = getattr(pooling, cfg.pooling)(self.auto_cfg)
        self.fc = nn.Linear(self.auto_cfg.hidden_size, 2)  # Target Class: content, wording
        if self.cfg.load_pretrained:
            self.model.load_state_dict(
                torch.load(cfg.checkpoint_dir + cfg.state_dict),
                strict=False
            )

        if cfg.reinit:
            self._init_weights(self.fc)
            reinit_topk(self.model, cfg.num_reinit)

        if cfg.freeze:
            freeze(self.model.embeddings)
            freeze(self.model.encoder.layer[:cfg.num_freeze])

        if cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()

    def _init_weights(self, module) -> None:
        """ over-ride initializes weights of the given module function (+initializes LayerNorm) """
        if isinstance(module, nn.Linear):
            if self.cfg.init_weight == 'normal':
                module.weight.data.normal_(mean=0.0, std=self.auto_cfg.initializer_range)
            elif self.cfg.init_weight == 'xavier_uniform':
                module.weight.data = nn.init.xavier_uniform_(module.weight.data)
            elif self.cfg.init_weight == 'xavier_normal':
                module.weight.data = nn.init.xavier_normal_(module.weight.data)
            elif self.cfg.init_weight == 'kaiming_uniform':
                module.weight.data = nn.init.kaiming_uniform_(module.weight.data)
            elif self.cfg.init_weight == 'kaiming_normal':
                module.weight.data = nn.init.kaiming_normal_(module.weight.data)
            elif self.cfg.init_weight == 'orthogonal':
                module.weight.data = nn.init.orthogonal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.auto_cfg.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            """ reference from torch.nn.Layernorm with elementwise_affine=True """
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def feature(self, inputs: dict):
        outputs = self.model(**inputs)
        return outputs

    def forward(self, inputs: dict) -> list[Tensor]:
        outputs = self.feature(inputs)
        feature = outputs.last_hidden_state
        if self.cfg.pooling == 'WeightedLayerPooling':
            feature = outputs.hidden_states
        embedding = self.pooling(feature, inputs['attention_mask'])
        logit = self.fc(embedding)
        return logit


class MaskedOneToOneModel(nn.Module):
    """
    Model Class for OneToOne Dataschema Pipeline,
    which is applied Masking Tensor for extracting only target token's embedding
    Args:
        cfg: configuration.CFG
    """
    def __init__(self, cfg: configuration.CFG) -> None:
        super().__init__()
        self.cfg = cfg
        self.auto_cfg = AutoConfig.from_pretrained(
            cfg.model,
            output_hidden_states=True
        )
        self.auto_cfg.attention_probs_dropout_prob = self.cfg.attention_probs_dropout_prob
        self.auto_cfg.hidden_dropout_prob = self.cfg.hidden_dropout_prob

        self.model = AutoModel.from_pretrained(
            cfg.model,
            config=self.auto_cfg
        )
        self.model.resize_token_embeddings(len(self.cfg.tokenizer))
        # self.pooling = getattr(pooling, cfg.pooling)(self.auto_cfg)
        self.fc = nn.Linear(self.auto_cfg.hidden_size, 2)  # Target Class: content, wording
        if self.cfg.load_pretrained:
            self.model.load_state_dict(
                torch.load(cfg.checkpoint_dir + cfg.state_dict),
                strict=False
            )

        if cfg.reinit:
            self._init_weights(self.fc)
            reinit_topk(self.model, cfg.num_reinit)

        if cfg.freeze:
            freeze(self.model.embeddings)
            freeze(self.model.encoder.layer[:cfg.num_freeze])

        if cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()

    def _init_weights(self, module) -> None:
        """ over-ride initializes weights of the given module function (+initializes LayerNorm) """
        if isinstance(module, nn.Linear):
            if self.cfg.init_weight == 'normal':
                module.weight.data.normal_(mean=0.0, std=self.auto_cfg.initializer_range)
            elif self.cfg.init_weight == 'xavier_uniform':
                module.weight.data = nn.init.xavier_uniform_(module.weight.data)
            elif self.cfg.init_weight == 'xavier_normal':
                module.weight.data = nn.init.xavier_normal_(module.weight.data)
            elif self.cfg.init_weight == 'kaiming_uniform':
                module.weight.data = nn.init.kaiming_uniform_(module.weight.data)
            elif self.cfg.init_weight == 'kaiming_normal':
                module.weight.data = nn.init.kaiming_normal_(module.weight.data)
            elif self.cfg.init_weight == 'orthogonal':
                module.weight.data = nn.init.orthogonal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.auto_cfg.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            """ reference from torch.nn.Layernorm with elementwise_affine=True """
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def feature(self, inputs: dict):
        outputs = self.model(**inputs)
        return outputs

    def forward(self, inputs: Dict) -> List[Tensor]:
        """ Remove pooling layer """
        outputs = self.feature(inputs)
        embedding = outputs.last_hidden_state
        logit = self.fc(embedding).squeeze(-1)
        return logit


class OneToManyModel(nn.Module):
    """
    Model Class for OneToMany Dataschema Pipeline
    This class apply reinit_top_k encoder's weight, init weights of fine-tune stage (fully-connect, regressor-head)
    And then, pooling each sub-sequence embedding which is meaning that each unique instance, this pipeline doesn't use pooling
    after calculate logit of each token embedding, we torch.mean(all of logit), So not need to pooling method
    Args:
        cfg: configuration.CFG
    """
    def __init__(self, cfg: configuration.CFG) -> None:
        super().__init__()
        self.cfg = cfg
        self.auto_cfg = AutoConfig.from_pretrained(
            cfg.model,
            output_hidden_states=True
        )
        self.auto_cfg.attention_probs_dropout_prob = self.cfg.attention_probs_dropout_prob
        self.auto_cfg.hidden_dropout_prob = self.cfg.hidden_dropout_prob

        self.model = AutoModel.from_pretrained(
            cfg.model,
            config=self.auto_cfg
        )
        self.model.resize_token_embeddings(len(self.cfg.tokenizer))
        # self.pooling = getattr(pooling, cfg.pooling)(self.auto_cfg)
        self.fc = nn.Linear(self.auto_cfg.hidden_size, 2)  # Target Class: content, wording
        if self.cfg.load_pretrained:
            self.model.load_state_dict(
                torch.load(cfg.checkpoint_dir + cfg.state_dict),
                strict=False
            )  # load student model's weight: it already has fc layer, so need to init fc layer later

        if cfg.reinit:
            self._init_weights(self.fc)
            reinit_topk(self.model, cfg.num_reinit)

        if cfg.freeze:
            freeze(self.model.embeddings)
            freeze(self.model.encoder.layer[:cfg.num_freeze])

        if cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()

    def _init_weights(self, module) -> None:
        """ over-ride initializes weights of the given module function (+initializes LayerNorm) """
        if isinstance(module, nn.Linear):
            if self.cfg.init_weight == 'normal':
                module.weight.data.normal_(mean=0.0, std=self.auto_cfg.initializer_range)
            elif self.cfg.init_weight == 'xavier_uniform':
                module.weight.data = nn.init.xavier_uniform_(module.weight.data)
            elif self.cfg.init_weight == 'xavier_normal':
                module.weight.data = nn.init.xavier_normal_(module.weight.data)
            elif self.cfg.init_weight == 'kaiming_uniform':
                module.weight.data = nn.init.kaiming_uniform_(module.weight.data)
            elif self.cfg.init_weight == 'kaiming_normal':
                module.weight.data = nn.init.kaiming_normal_(module.weight.data)
            elif self.cfg.init_weight == 'orthogonal':
                module.weight.data = nn.init.orthogonal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.auto_cfg.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            """ reference from torch.nn.Layernorm with elementwise_affine=True """
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def feature(self, inputs: Dict):
        outputs = self.model(**inputs)
        return outputs

    def forward(self, inputs: Dict) -> List[Tensor]:
        """ Remove pooling layer """
        outputs = self.feature(inputs)
        embedding = outputs.last_hidden_state
        logit = self.fc(embedding).squeeze(-1)
        return logit
