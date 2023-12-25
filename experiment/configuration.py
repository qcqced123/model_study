import torch
from transformers import AutoTokenizer
from utils.helper import select_model_file

class CFG:
    """ Pipeline Setting """
    train, test = True, False
    checkpoint_dir = './saved/model'
    resume, load_pretrained,  state_dict = True, False, '/'
    name = 'DeBERTa_MLM'
    loop = 'mpl_loop'
    dataset = 'FBPDataset'  # dataset_class.dataclass.py -> FBPDataset, MPLDataset
    arch_name = 'attention'
    model_name = 'deberta'
    module_name = 'DeBERTa'
    model = select_model_file(arch=arch_name, model=model_name)
    tmp_model = 'microsoft/deberta-v3-large'
    tokenizer = AutoTokenizer.from_pretrained(tmp_model)
    pooling = 'MeanPooling'  # mean, attention, max, weightedlayer, concat, conv1d, lstm

    """ Common Options """
    wandb = True
    optuna = False
    seed = 42
    cfg_name = 'CFG'
    n_gpu = 1
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gpu_id = 0
    num_workers = 0

    """ Data Options """
    n_folds = 5
    max_len = 512
    epochs = 180
    batch_size = 64
    smart_batch = False

    """ Gradient Options """
    amp_scaler = False
    gradient_checkpoint = True  # save parameter
    clipping_grad = True  # clip_grad_norm
    n_gradient_accumulation_steps = 1
    max_grad_norm = 1000

    """ Loss & Metrics Options """
    loss_fn = 'SmoothL1Loss'
    val_loss_fn = 'WeightedMSELoss'
    reduction = 'mean'
    content_weight = 0.45
    wording_weight = 0.55
    metrics = ['MCRMSE', 'f_beta', 'recall']

    """ Optimizer with LLRD Options """
    optimizer = 'AdamW'  # options: SWA, AdamW
    llrd = True
    layerwise_lr = 5e-6
    layerwise_lr_decay = 0.9
    layerwise_weight_decay = 1e-2
    layerwise_adam_epsilon = 1e-6
    layerwise_use_bertadam = False
    betas = (0.9, 0.999)

    """ Scheduler Options """
    scheduler = 'cosine_annealing'  # options: cosine, linear, cosine_annealing, linear_annealing
    batch_scheduler = True
    num_cycles = 0.5  # num_warmup_steps = 0
    warmup_ratio = 0.1  # options: 0.05, 0.1

    """ SWA Options """
    swa = True
    swa_start = 2
    swa_lr = 1e-4
    anneal_epochs = 4
    anneal_strategy = 'cos'  # default = cos, available option: linear

    """ Model Options """
    vocab_size = tokenizer.vocab_size
    max_seq = 512
    num_layers = 12
    num_emd = 2
    num_attention_heads = 12
    dim_model = 768
    dim_ffn = 2048
    hidden_act = 'gelu'
    layer_norm_eps = 1e-7
    attention_probs_dropout_prob = 0.1
    hidden_dropout_prob = 0.1
    init_weight = 'orthogonal'  # options: normal, xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal
    initializer_range = 0.02
    stop_mode = 'min'
    freeze = False
    num_freeze = 2
    reinit = True
    num_reinit = 0
    awp = False
    nth_awp_start_epoch = 10
    awp_eps = 1e-2
    awp_lr = 1e-4


if __name__ == "__main__":
    test = CFG
    print(test.model)
