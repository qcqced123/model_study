import numpy as np
import torch
import transformers
import more_itertools
from torch.utils.data import Sampler, Dataset, DataLoader
from torch import Tensor
from typing import List, Tuple, Dict
from utils.helper import seed_worker


def get_optimizer_grouped_parameters(model, layerwise_lr, layerwise_weight_decay, layerwise_lr_decay):
    """ Grouped Version: Layer-wise learning rate decay """
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # initialize lr for task specific layer
    optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if "model" not in n],
                                     "weight_decay": 0.0,
                                     "lr": layerwise_lr,
                                     }, ]
    # initialize lrs for every layer
    layers = [model.model.embeddings] + list(model.model.encoder.layer)
    layers.reverse()
    lr = layerwise_lr
    for layer in layers:
        optimizer_grouped_parameters += [
            {"params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": layerwise_weight_decay,
             "lr": lr,
             },
            {"params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0,
             "lr": lr,
             },]
        lr *= layerwise_lr_decay
    return optimizer_grouped_parameters


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if "model" not in n],
         'lr': decoder_lr, 'weight_decay': 0.0}
    ]
    return optimizer_parameters


def collate(inputs) -> Dict:
    """ Descending sort inputs by length of sequence """
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]
    return inputs


def one2many_collate(inputs, label1, label2) -> Tuple[Dict, Tensor, Tensor]:
    """ Descending sort inputs by length of sequence """
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]
    label1, label2 = label1[:, :mask_len], label2[:, :mask_len]
    return inputs, label1, label2


def get_dataloader(cfg, dataset: Dataset, generator: torch.Generator, shuffle: bool = True, collate_fn=None, sampler=None, drop_last: bool = True) -> DataLoader:
    """ function for initializing torch.utils.data.DataLoader Module """
    dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            sampler=sampler,
            worker_init_fn=seed_worker,
            generator=generator,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=drop_last,
        )
    return dataloader


def get_swa_scheduler(cfg, optimizer):
    """  SWA Scheduler """
    swa_scheduler = getattr(torch.optim.swa_utils, 'SWALR')(
        optimizer,
        swa_lr=cfg.swa_lr,
        anneal_epochs=cfg.anneal_epochs,
        anneal_strategy=cfg.anneal_strategy
    )
    return swa_scheduler


def get_scheduler(cfg, optimizer, len_train: int):
    """ Select Scheduler Function """
    scheduler_dict = {
        'cosine_annealing': 'get_cosine_with_hard_restarts_schedule_with_warmup',
        'cosine': 'get_cosine_schedule_with_warmup',
        'linear': 'get_linear_schedule_with_warmup'
    }
    lr_scheduler = getattr(transformers, scheduler_dict[cfg.scheduler])(
        optimizer,
        num_warmup_steps=int(len_train/cfg.batch_size * cfg.epochs/cfg.n_gradient_accumulation_steps) * cfg.warmup_ratio,
        num_training_steps=int(len_train/cfg.batch_size * cfg.epochs/cfg.n_gradient_accumulation_steps),
        num_cycles=cfg.num_cycles
    )
    return lr_scheduler


def get_name(cfg) -> str:
    """ get name of model """
    try:
        name = cfg.model.replace('/', '-')
    except ValueError:
        name = cfg.model
    return name


class SmartBatchingSampler(Sampler):
    """
    SmartBatching Sampler with naive pytorch torch.utils.data.Sampler implementation, which is iterable-style dataset
    not map-style dataset module.
    This class chunk whole of batch instances by length for making mini-batch by its length
    Args:
        data_instance: whole of batch instances, not mini-batch-level
        batch_size: amount of mini-batch from configuration.py or cfg.json
    Reference:
        https://www.kaggle.com/code/rhtsingh/speeding-up-transformer-w-optimization-strategies
    """
    def __init__(self, data_instance: Tensor, batch_size: int) -> None:
        super(SmartBatchingSampler, self).__init__(data_instance)
        self.len = len(data_instance)
        sample_lengths = [len(seq) for seq in data_instance]
        argsort_inds = np.argsort(sample_lengths)
        self.batches = list(more_itertools.chunked(argsort_inds, n=batch_size))
        self._backsort_inds = None

    def __iter__(self) -> int:
        if self.batches:
            last_batch = self.batches.pop(-1)
            np.random.shuffle(self.batches)
            self.batches.append(last_batch)
        self._inds = list(more_itertools.flatten(self.batches))
        yield from self._inds

    def __len__(self) -> int:
        return self.len

    @property
    def backsort_inds(self):
        if self._backsort_inds is None:
            self._backsort_inds = np.argsort(self._inds)
        return self._backsort_inds


class SmartBatchingCollate:
    """
    SmartBatchingCollate will add padding upto highest sequence length, make attention masks,
    targets for each sample in batch.
    Args:
        labels: target value of training dataset
        max_length: value from configuration.py or cfg.json which is initialized by user
        pad_token_id: int value from pretrained tokenizer, AutoTokenizer.tokenizer.pad_token_ids
    Reference:
        https://www.kaggle.com/code/rhtsingh/speeding-up-transformer-w-optimization-strategies
    """
    def __init__(self, labels: Tensor, max_length: int, pad_token_id: int) -> None:
        self._labels = labels
        self._max_length = max_length
        self._pad_token_id = pad_token_id

    def __call__(self, batch: Tuple) -> Tuple[Tensor, Tensor]:
        if self._labels is not None:
            sequences, targets = list(zip(*batch))
        else:
            sequences = list(batch)

        input_ids, attention_mask = self.pad_sequence(
            sequences,
            max_sequence_length=self._max_length,
            pad_token_id=self._pad_token_id
        )

        output = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if self._labels is not None:
            labels = torch.as_tensor(targets)
        else:
            output = input_ids, attention_mask
        return output, labels

    @staticmethod
    def pad_sequence(sequence_batch: List, max_sequence_length: int, pad_token_id: int) -> Tuple[Tensor, Tensor]:
        max_batch_len = max(len(sequence) for sequence in sequence_batch)
        max_len = min(max_batch_len, max_sequence_length)
        padded_sequences, attention_masks = [[] for i in range(2)]
        attend, no_attend = 1, 0
        for sequence in sequence_batch:
            # As discussed above, truncate if exceeds max_len
            new_sequence = list(sequence[:max_len])

            attention_mask = [attend] * len(new_sequence)
            pad_length = max_len - len(new_sequence)

            new_sequence.extend([pad_token_id] * pad_length)
            attention_mask.extend([no_attend] * pad_length)

            padded_sequences.append(new_sequence)
            attention_masks.append(attention_mask)
        padded_sequences = torch.as_tensor(padded_sequences)
        attention_masks = torch.as_tensor(attention_masks)
        return padded_sequences, attention_masks


class MiniBatchCollate(object):
    """
    Collate class for torch.utils.data.DataLoader
    This class object to use variable data such as NLP text sequence
    If you use static padding with AutoTokenizer, you don't need this class object
    But if you use dynamic padding with AutoTokenizer, you must use this class object & call
    Args:
        batch: data instance from torch.utils.data.DataSet
    """
    def __init__(self, batch: torch.utils.data.DataLoader) -> None:
        self.batch = batch

    def __call__(self) -> tuple[dict[Tensor, Tensor, Tensor], Tensor, Tensor]:
        inputs, labels, position_list = self.batch
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-1
        )
        position_list = torch.nn.utils.rnn.pad_sequence(
            position_list,
            batch_first=True,
            padding_value=-1
        )
        return inputs, labels, position_list


class AWP:
    """
    Adversarial Weight Perturbation for OneToOne Trainer
    References:
        https://www.kaggle.com/code/skraiii/pppm-tokenclassificationmodel-train-8th-place
    """
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        awp: bool,
        adv_param: str = "weight",
        adv_lr: float=1.0,
        adv_eps: float=0.01
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.awp = awp
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, inputs: Dict, c_label: Tensor, w_label: Tensor):
        with torch.cuda.amp.autocast(enabled=self.awp):
            self._save()
            self._attack_step()
            pred_list = self.model(inputs)
            c_pred, w_pred = pred_list[:, 0], pred_list[:, 1]
            c_adv_loss, w_adv_loss = self.criterion(c_pred, c_label), self.criterion(w_pred, w_label)
            # mask = (label.view(-1, 1) != -1)  # this line will be needed for OneToMany Trainer
            # adv_loss = torch.masked_select(adv_loss, mask).mean()  # this line will be needed for OneToMany Trainer
            self.optimizer.zero_grad()
        return c_adv_loss, w_adv_loss

    def _attack_step(self) -> None:
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(
                            param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )

    def _save(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping(object):
    """
    Monitor a metric and stop training when it stops improving.

    Args:
        mode: 'min' for loss base val_score for loss, 'max' for metric base val_score
        patience: number of checks with no improvement, default = 3
        min_delta: minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute
            change of less than or equal to `min_delta`, will count as no improvement. default = 0.0
        detect_anomaly: When set ``True``, stops training when the monitor becomes NaN or infinite, etc
                        default = True
    """
    def __init__(self, mode: str, patience: int = 3, min_delta: float = 0.0, detect_anomaly: bool = True):
        self.mode = mode
        self.early_stop = False
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.detect_anomaly = detect_anomaly
        self.val_score = -np.inf

        if self.mode == 'min':
            self.val_score = np.inf

    def detecting_anomaly(self) -> None:
        """ Detecting Trainer's Error and Stop train loop """
        torch.autograd.set_detect_anomaly(self.detect_anomaly)
        return

    def __call__(self, score: any) -> None:
        """ When call by Trainer Loop, Check Trainer need to early stopping """
        if self.mode == 'min':
            if self.val_score >= score:
                self.counter = 0
                self.val_score = score
            else:
                self.counter += 1

        if self.mode == 'max':
            if score >= self.val_score:
                self.counter = 0
                self.val_score = score
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            print('Early STOP')
