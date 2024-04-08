import importlib.util
import torch
import torch.nn as nn
from peft import PeftType, TaskType
from peft import get_peft_config, get_peft_model, LoraConfig, replace_lora_weights_loftq
from peft import PromptEncoderConfig, PromptEncoder
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig
from typing import Tuple
from configuration import CFG


class AbstractTask:
    """ Abstract model class for all tasks in this project
    Each task should inherit this class for using common functionalities

    Functions:
        1) Init Gradient Checkpointing Flag
        2) Weight Initialization
            - Pytorch Default Weight Initialization: He Initialization (Kaiming Initialization)
        3) Interface method for making model instance in runtime
        4) Apply fine-tune options, which are selected in configuration.json
            - load pretrained weights for fine-tune (own your hub, huggingface model hub ... etc)
            - apply PEFT (Quantization, LoRA, QLoRA, P-Tuning, ...)
    """
    def __init__(self, cfg: CFG) -> None:
        super(AbstractTask, self).__init__()
        self.cfg = cfg

    def _init_weights(self, module: nn.Module) -> None:
        """ Over-ride initializes weights of the given module function for torch models
        you must implement this function in your task class

        Args:
            module (:obj:`torch.nn.Module`):
                The module to initialize weights for
        """
        """ over-ride initializes weights of the given module function (+initializes LayerNorm) """
        if isinstance(module, nn.Linear):
            if self.cfg.init_weight == 'normal':
                module.weight.data.normal_(mean=0.0, std=self.cfg.initializer_range)
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
            module.weight.data.normal_(mean=0.0, std=self.cfg.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            """ reference from torch.nn.Layernorm with elementwise_affine=True """
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def select_model(self, num_layers: int) -> nn.Module:
        """ Selects architecture for each task (pretrain),
        you can easily select your model for experiment from json config files

        1) select .py file from input config settings
        2) select class object from input config settings

        Args:
            num_layers:
                The number of layers for each task

        Returns:
            model (:obj:`nn.Module`):
                The model to use for each task
        """
        base_path = 'experiment/models/'
        arch_path = f"{base_path + self.cfg.arch_name + '/' + self.cfg.model_name}.py"
        spec = importlib.util.spec_from_file_location(self.cfg.model_name, arch_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = getattr(module, self.cfg.module_name)(self.cfg, num_layers)  # get instance in runtime
        return model

    def select_pt_model(self) -> Tuple[nn.Module, nn.Module]:
        """ Selects architecture for each task (fine-tune),
        you can easily select your model for experiment from json config files
        or choose the pretrained weight hub (own your pretrain, huggingface ... etc)

        Var:
            self.hub: hub name for choosing pretrained weights, now you can ONLY select huggingface hub,
                      because LoRA, QLoRA, P-Tuning are not yet implemented for local hub models
        Notes:
            pass this arguments => 'huggingface', 'local' (Not Yet)

        Reference:
            https://huggingface.co/docs/peft/en/developer_guides/quantization
            https://github.com/huggingface/peft/tree/main/examples/loftq_finetuning
        """
        model = None,
        bit_config = None
        prompt_encoder = None
        if self.cfg.qlora:
            bit_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 is recommended
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type='nf4'
            )
        if self.cfg.hub == 'local': raise NotImplementedError("Not Yet, Please pass hub argument to huggingface for now")
        elif self.cfg.hub == 'huggingface':
            config = AutoConfig.from_pretrained(self.cfg.model_name)
            model = AutoModel.from_pretrained(
                self.cfg.model_name,
                config=config,
                quantization_config=bit_config
            )

        model = self.apply_peft_lora(model) if self.cfg.lora else self.apply_peft_qlora(model)
        if self.cfg.p_tuning: prompt_encoder = self.apply_peft_prompt_tuning(model)
        return model, prompt_encoder

    def apply_peft_lora(self, model: nn.Module) -> nn.Module:
        """ class method for applying peft lora to pretrained model in fine-tune stage

        Args:
            model: pretrained model from huggingface model hub

        Reference:
            https://github.com/huggingface/peft?tab=readme-ov-file
            https://huggingface.co/docs/peft/en/developer_guides/lora
            https://arxiv.org/abs/2106.09685
            https://arxiv.org/abs/2305.14314
        """
        lora_config = LoraConfig(
            task_type=getattr(TaskType, self.cfg.task_type),
            inference_mode=False,
            r=self.cfg.lora_rank,  # rank value
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout
        )
        return get_peft_model(
            model=model,
            peft_config=lora_config,
        )

    def apply_peft_qlora(self, model: nn.Module) -> nn.Module:
        """ class method for applying peft Q-lora to pretrained model in fine-tune stage

        Args:
            model: pretrained model from huggingface model hub

        Reference:
            https://github.com/huggingface/peft?tab=readme-ov-file
            https://huggingface.co/docs/peft/en/developer_guides/lora
            https://arxiv.org/abs/2106.09685
            https://arxiv.org/abs/2305.14314
        """
        lora_config = LoraConfig(
            task_type=getattr(TaskType, self.cfg.task_type),
            inference_mode=False,
            r=self.cfg.lora_rank,  # rank value
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout
        )
        return get_peft_model(
            model=model,
            peft_config=lora_config,
        )

    def apply_peft_prompt_tuning(self) -> nn.Module:
        """ class method for applying peft p-tuning to pretrained model in fine-tune stage

        Args:
            model: pretrained model from huggingface model hub
        """
        config = PromptEncoderConfig(
            peft_type=self.cfg.prompt_tuning_type,
            task_type=self.cfg.task_type,
            num_virtual_tokens=self.cfg.num_virtual_tokens,
            token_dim=self.cfg.virtual_token_dim,
            encoder_reparameterization_type="MLP",
            encoder_hidden_size=self.cfg.prompt_encoder_hidden_size,
        )
        prompt_encoder = PromptEncoder(config)
        return prompt_encoder
