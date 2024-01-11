import torch
from transformers import AutoTokenizer


class CFG:
    """ Base Configuration Class for various experiments
    This Module has all of hyper-parameters for whole this project such as training, model, data, optimizer, scheduler, loss, metrics, ... etc
    You can change hyper-parameters easily by changing json file in configuration data_folder
    Each json file has same name with model name, so you can easily find hyper-parameters for each model
    And then, individual JSON files adjust only the hyperparameters needed for individual experiments
    """

    """ Pipeline Setting """
    train, test = True, False
    checkpoint_dir = 'saved/model'
    resume, load_pretrained, state_dict = True, False, '/'
    generator_state_dict = '/'
    discriminator_state_dict = '/'
    name = 'MaskedLanguageModel'
    datafolder = 'wikipedia_en'
    trainer = 'PreTrainTuner'
    loop = 'train_loop'
    hf_dataset = 'wikimedia/wikipedia'
    language = '20231101.en'
    dataset = 'MLMDataset'  # dataset_class.dataclass.py -> MLMDataset, CLMDataset ... etc
    arch_name = 'attention'
    model_name = 'deberta'
    module_name = 'BERT'
    tokenizer_name = 'microsoft/deberta-v3-large'  # later, remove this line
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    task = 'MaskedLanguageModel'  # options: MaskedLanguageModel, CasualLanguageModel
    pooling = 'MeanPooling'

    """ Common Options """
    wandb = True
    optuna = False
    cfg_name = 'CFG'
    seed = 42
    n_gpu = 1
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gpu_id = 0
    num_workers = 4

    """ Data Options """
    split_ratio = 0.2
    n_folds = 5
    max_len = 512
    epochs = 10
    batch_size = 64
    smart_batch = False
    val_check = 1000  # setting for validation check frequency (unit: step)

    """ Gradient Options """
    amp_scaler = True
    gradient_checkpoint = True
    clipping_grad = True
    n_gradient_accumulation_steps = 1
    max_grad_norm = 1

    """ Loss & Metrics Options """
    loss_fn = 'CrossEntropyLoss'
    val_loss_fn = 'CrossEntropyLoss'
    reduction = 'mean'
    metrics = ['accuracy', 'precision', 'recall']

    """ Optimizer with LLRD Options """
    optimizer = 'AdamW'  # options: SWA, AdamW
    llrd = True
    layerwise_lr = 5e-5
    layerwise_lr_decay = 0.9
    layerwise_weight_decay = 1e-2
    layerwise_adam_epsilon = 1e-6
    layerwise_use_bertadam = False
    lr = 5e-5
    weight_decay = 1e-2
    adam_epsilon = 1e-6
    use_bertadam = False
    betas = (0.9, 0.999)

    """ Scheduler Options """
    scheduler = 'cosine_annealing'  # options: cosine, linear, cosine_annealing, linear_annealing
    batch_scheduler = True
    num_cycles = 0.5  # num_warmup_steps = 0
    warmup_ratio = 0.1  # if you select per step, you should consider size of epoch

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
    dim_ffn = 3072
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

    """ Pre-Training Option """
    is_mf_embedding = False
    mlm_masking = 'SubWordMasking'
    mlm_probability = 0.15

    # 2) RTD Option
    rtd_masking = 'SubWordMasking'
    generator = 'DeBERTa'
    discriminator = 'DeBERTa'
    is_share_embed = True
    share_embed_method = 'ES'  # options: GDES (Gradient Disentangled Embedding Space)
    is_generator_resume = False  # load pretrained generator's weight
    is_discriminator_resume = False  # load pretrained discriminator's weight
    generator_load_pretrained = False
    discriminator_load_pretrained = False
    discriminator_lambda = 10
    generator_num_layers = 6
    discriminator_num_layers = 12

    # 3) SBO Option
    span_encoder_name = 'DeBERTa'
    masking_budget = 0.15
    span_probability = 0.2
    max_span_length = 10
    is_concatenate = True
