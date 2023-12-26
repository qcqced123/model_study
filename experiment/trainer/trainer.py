import gc
import torch
import numpy as np
import pandas as pd
import transformers

from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel

from tqdm.auto import tqdm
from torch import Tensor
from typing import Tuple

import experiment.dataset_class.dataclass as dataset_class
from tuner import mlm, clm
from losses import loss
from experiment.model import model as model_arch
from experiment.configuration import CFG
from experiment.dataset_class.preprocessing import load_pkl
from experiment.trainer.trainer_utils import get_optimizer_grouped_parameters, get_scheduler, collate
from experiment.trainer.trainer_utils import AverageMeter, AWP, get_dataloader, get_swa_scheduler


class PreTrainTuner:
    """ Trainer class for Pre-Train Pipeline, such as MLM, CLM ... etc
    So, if you want set options, go to cfg.json file or configuration.py
    Args:
        cfg: configuration module, configuration.py
        generator: torch.Generator, for init pytorch random seed
    """
    def __init__(self, cfg: CFG, generator: torch.Generator) -> None:
        self.cfg = cfg
        self.model_name = self.cfg.module_name
        self.tokenizer = self.cfg.tokenizer
        self.generator = generator

    def make_batch(self) -> Tuple[DataLoader, DataLoader, int]:
        """ Function for making batch instance """
        train = load_pkl(f'./dataset_class/data_folder/{self.cfg.datafolder}/train')
        valid = load_pkl(f'./dataset_class/data_folder/{self.cfg.datafolder}/valid')

        # 1) Custom Datasets
        train_dataset = getattr(dataset_class, self.cfg.dataset)(train)
        valid_dataset = getattr(dataset_class, self.cfg.dataset)(valid)

        # 2) Custom Collator
        collate_fn = None
        if self.cfg.task == 'MaskedLanguageModel':
            collate_fn = getattr(mlm, 'MLMCollator')(self.cfg)
        elif self.cfg.task == 'CasualLanguageModel':
            collate_fn = getattr(clm, 'CLMCollator')(self.cfg)

        # 3) Initializing torch.utils.data.DataLoader Module
        loader_train = get_dataloader(
            cfg=self.cfg,
            dataset=train_dataset,
            collate_fn=collate_fn,
            generator=self.generator
        )
        loader_valid = get_dataloader(
            cfg=self.cfg,
            dataset=valid_dataset,
            collate_fn=collate_fn,
            generator=self.generator,
            shuffle=False,
            drop_last=False
        )
        return loader_train, loader_valid, len(train['input_ids'])

    def model_setting(self, len_train: int):
        """ Function for init backbone's configuration & train utils setting,
        the design is inspired by the Builder pattern.
        """
        model = getattr(model_arch, self.cfg.task)(self.cfg)
        # load checkpoint when you set 'resume' to True
        if self.cfg.resume:
            model.load_state_dict(torch.load(self.cfg.checkpoint_dir + self.cfg.state_dict))
        model.to(self.cfg.device)

        criterion = getattr(loss, self.cfg.loss_fn)(self.cfg.reduction)
        val_criterion = getattr(loss, self.cfg.val_loss_fn)(self.cfg.reduction)
        grouped_optimizer_params = get_optimizer_grouped_parameters(
            model,
            self.cfg.layerwise_lr,
            self.cfg.layerwise_weight_decay,
            self.cfg.layerwise_lr_decay
        )
        optimizer = getattr(transformers, self.cfg.optimizer)(
            params=grouped_optimizer_params,
            lr=self.cfg.layerwise_lr,
            eps=self.cfg.layerwise_adam_epsilon,
            correct_bias=not self.cfg.layerwise_use_bertadam
        )
        lr_scheduler = get_scheduler(self.cfg, optimizer, len_train)

        # init SWA Module
        swa_model, swa_scheduler = None, None
        if self.cfg.swa:
            swa_model = AveragedModel(model)
            swa_scheduler = get_swa_scheduler(self.cfg, optimizer)

        # init AWP Module
        awp = None
        if self.cfg.awp:
            awp = AWP(
                model,
                criterion,
                optimizer,
                self.cfg.awp,
                adv_lr=self.cfg.awp_lr,
                adv_eps=self.cfg.awp_eps
            )
        return model, criterion, val_criterion, optimizer, lr_scheduler, awp, swa_model, swa_scheduler

    def train_fn(self, loader_train, model, criterion, optimizer, scheduler, epoch, awp=None, swa_model=None, swa_start=None, swa_scheduler=None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """ function for train loop """
        # torch.autograd.set_detect_anomaly(True)
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp_scaler)
        losses, c_losses, w_losses = AverageMeter(), AverageMeter(), AverageMeter()
        model.train()
        for step, (inputs, labels) in enumerate(tqdm(loader_train)):
            optimizer.zero_grad()
            inputs = collate(inputs)  # need to check this method, when applying smart batching

            for k, v in inputs.items():
                inputs[k] = v.to(self.cfg.device)  # train to gpu
            labels = labels.to(self.cfg.device)  # Two target values to GPU
            label_content, label_wording = labels[:, 0], labels[:, 1]
            batch_size = labels.size(0)

            with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
                pred_list = model(inputs)
                c_pred, w_pred = pred_list[:, 0], pred_list[:, 1]
                c_loss, w_loss = criterion(c_pred, label_content), criterion(w_pred, label_wording)
                # losses = (self.cfg.content_weight * c_loss) + (self.cfg.wording_weight * w_loss)  # Weighted MCRMSE Loss
                loss = (self.cfg.wording_weight * w_loss)  # wording only train

            if self.cfg.n_gradient_accumulation_steps > 1:
                loss = loss / self.cfg.n_gradient_accumulation_steps

            scaler.scale(loss).backward()
            losses.update(loss.detach().cpu().numpy(), batch_size)  # Must do detach() for avoid memory leak
            c_losses.update(c_loss.detach().cpu().numpy(), batch_size)
            w_losses.update(w_loss.detach().cpu().numpy(), batch_size)

            if self.cfg.awp and epoch >= self.cfg.nth_awp_start_epoch:
                awp_c_loss, awp_w_loss = awp.attack_backward(inputs, label_content, label_wording)
                scaler.scale(self.cfg.content_weight*awp_c_loss + self.cfg.wording_weight*awp_w_loss).backward()
                awp._restore()

            if self.cfg.clipping_grad and (step + 1) % self.cfg.n_gradient_accumulation_steps == 0 or self.cfg.n_gradient_accumulation_steps == 1:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm(
                    model.parameters(),
                    self.cfg.max_grad_norm * self.cfg.n_gradient_accumulation_steps
                )
                # Stochastic Weight Averaging
                if epoch >= int(swa_start):
                    swa_model.update_parameters(model)
                    swa_scheduler.step()

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            del inputs, labels, label_content, label_wording, pred_list, c_loss, w_loss, loss
            torch.cuda.empty_cache()
            gc.collect()

        grad_norm = grad_norm.detach().cpu().numpy()
        return losses.avg, c_losses.avg, w_losses.avg, grad_norm, scheduler.get_lr()[0]

    def valid_fn(self, loader_valid, model, val_criterion) -> Tuple[Tensor, Tensor, Tensor]:
        """ function for validation loop """
        valid_losses, c_losses, w_losses = AverageMeter(), AverageMeter(), AverageMeter()
        model.eval()
        with torch.no_grad():
            for step, (inputs, labels) in enumerate(tqdm(loader_valid)):
                inputs = collate(inputs)  # need to check this method, when applying smart batching
                for k, v in inputs.items():
                    inputs[k] = v.to(self.cfg.device)
                labels = labels.to(self.cfg.device)
                label_content, label_wording = labels[:, 0], labels[:, 1]
                batch_size = labels.size(0)

                pred_list = model(inputs)
                c_pred, w_pred = pred_list[:, 0], pred_list[:, 1]
                c_loss, w_loss = val_criterion(c_pred, label_content), val_criterion(w_pred, label_wording)
                # losses = (c_loss + w_loss) / 2  # compute mc rmse
                loss = w_loss  # compute only for 1 target

                valid_losses.update(loss.detach().cpu().numpy(), batch_size)
                c_losses.update(c_loss.detach().cpu().numpy(), batch_size)
                w_losses.update(w_loss.detach().cpu().numpy(), batch_size)

            del inputs, labels, label_content, label_wording, pred_list, c_loss, w_loss, loss
            torch.cuda.empty_cache()
            gc.collect()
        return valid_losses.avg, c_losses.avg, w_losses.avg

    def swa_fn(self, loader_valid, swa_model, val_criterion):
        """ Stochastic Weight Averaging, it consumes more GPU VRAM & training times """
        swa_model.eval()
        valid_losses, c_losses, w_losses = AverageMeter(), AverageMeter(), AverageMeter()
        with torch.no_grad():
            for step, (inputs, labels) in enumerate(tqdm(loader_valid)):
                inputs = collate(inputs)
                for k, v in inputs.items():
                    inputs[k] = v.to(self.cfg.device)
                labels = labels.to(self.cfg.device)
                label_content, label_wording = labels[:, 0], labels[:, 1]
                batch_size = labels.size(0)

                pred_list = swa_model(inputs)
                c_pred, w_pred = pred_list[:, 0], pred_list[:, 1]
                c_loss, w_loss = val_criterion(c_pred, label_content), val_criterion(w_pred, label_wording)
                # losses = (c_loss + w_loss) / 2  # compute mc rmse
                loss = w_loss  # compute only for 1 target

                valid_losses.update(loss.detach().cpu().numpy(), batch_size)
                c_losses.update(c_loss.detach().cpu().numpy(), batch_size)
                w_losses.update(w_loss.detach().cpu().numpy(), batch_size)

            del inputs, labels, label_content, label_wording, pred_list, c_loss, w_loss, loss
            torch.cuda.empty_cache()
            gc.collect()
        return valid_losses.avg, c_losses.avg, w_losses.avg
