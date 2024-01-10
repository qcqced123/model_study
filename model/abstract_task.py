import importlib.util
import torch.nn as nn
from configuration import CFG


class AbstractTask:
    """ Abstract model class for all tasks in this project
    Each task should inherit this class for using common functionalities
    Functions:
        1) Init Gradient Checkpointing Flag
        2) Weight Initialization
            - Pytorch Default Weight Initialization: He Initialization (Kaiming Initialization)
        3) Interface method for making model instance in runtime
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
        """ Selects architecture for each task,
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

