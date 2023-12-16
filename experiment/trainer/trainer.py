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

import dataset_class.dataclass as dataset_class
from experiment.model import model as model_loss, model as model_arch
from configuration import CFG
from dataset_class.preprocessing import load_data
from trainer.trainer_utils import get_optimizer_grouped_parameters, get_scheduler, collate, one2many_collate
from trainer.trainer_utils import AverageMeter, AWP, get_dataloader, get_swa_scheduler


class OneToOneTrainer:
    """
    Trainer class for OneToOne DataSchema Pipeline, having many optimization options
    So, if you want set options, go to cfg.json file or configuration.py
    Args:
        cfg: configuration module, configuration.py
        generator: torch.Generator, for init pytorch random seed
    """
    def __init__(self, cfg: CFG, generator: torch.Generator) -> None:
        self.cfg = cfg
        self.model_name = self.cfg.model.split('/')[1]
        self.generator = generator
        self.p_df = load_data('./dataset_class/data_folder/one2one_prompt_df.csv')
        self.s_df = load_data('./dataset_class/data_folder/fold4_one2one_summaries_df.csv')
        self.tokenizer = self.cfg.tokenizer

    def make_batch(self, fold: int) -> tuple[DataLoader, DataLoader, pd.DataFrame]:
        """ function for making batch instance for random sampler, smart-batch sampler """
        train = self.s_df[self.s_df['fold'] != fold].reset_index(drop=True)
        valid = self.s_df[self.s_df['fold'] == fold].reset_index(drop=True)

        # 1) Custom Datasets
        train_dataset = getattr(dataset_class, self.cfg.dataset)(
            self.cfg, self.p_df, train
        )
        valid_dataset = getattr(dataset_class, self.cfg.dataset)(
            self.cfg, self.p_df, valid
        )
        # 2) Initializing torch.utils.data.DataLoader Module
        loader_train = get_dataloader(self.cfg, train_dataset, self.generator)
        loader_valid = get_dataloader(self.cfg, valid_dataset, self.generator, shuffle=False, drop_last=False)
        return loader_train, loader_valid, train

    def model_setting(self, len_train: int):
        """ function for init backbone's configuration & train utils setting """
        model = getattr(model_arch, self.cfg.model_arch)(self.cfg)
        # load checkpoint when you set 'resume' to True
        if self.cfg.resume:
            model.load_state_dict(torch.load(self.cfg.checkpoint_dir + self.cfg.state_dict))
        model.to(self.cfg.device)

        criterion = getattr(model_loss, self.cfg.loss_fn)(self.cfg.reduction)
        val_criterion = getattr(model_loss, self.cfg.val_loss_fn)(self.cfg.reduction)
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
                # loss = (self.cfg.content_weight * c_loss) + (self.cfg.wording_weight * w_loss)  # Weighted MCRMSE Loss
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
                # loss = (c_loss + w_loss) / 2  # compute mc rmse
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
                # loss = (c_loss + w_loss) / 2  # compute mc rmse
                loss = w_loss  # compute only for 1 target

                valid_losses.update(loss.detach().cpu().numpy(), batch_size)
                c_losses.update(c_loss.detach().cpu().numpy(), batch_size)
                w_losses.update(w_loss.detach().cpu().numpy(), batch_size)

            del inputs, labels, label_content, label_wording, pred_list, c_loss, w_loss, loss
            torch.cuda.empty_cache()
            gc.collect()
        return valid_losses.avg, c_losses.avg, w_losses.avg


class MaskedOneToOneTrainer:
    """
    Trainer class for OneToOne DataSchema Pipeline, having many optimization options
    So, if you want set options, go to cfg.json file or configuration.py
    And then, apply masking for extract embedding only target text
    Args:
        cfg: configuration module, configuration.py
        generator: torch.Generator, for init pytorch random seed
    """
    def __init__(self, cfg: CFG, generator: torch.Generator) -> None:
        self.cfg = cfg
        self.model_name = self.cfg.model.split('/')[1]
        self.generator = generator
        self.p_df = load_data('./dataset_class/data_folder/one2one_prompt_df.csv')
        self.s_df = load_data('./dataset_class/data_folder/7folds_one2one_df.csv')
        self.tokenizer = self.cfg.tokenizer

    def make_batch(self, fold: int) -> tuple[DataLoader, DataLoader, pd.DataFrame]:
        """ function for making batch instance for random sampler, smart-batch sampler """
        train = self.s_df[self.s_df['fold'] != fold].reset_index(drop=True)
        valid = self.s_df[self.s_df['fold'] == fold].reset_index(drop=True)

        # 1) Custom Datasets
        train_dataset = getattr(dataset_class, self.cfg.dataset)(
            self.cfg, self.p_df, train
        )
        valid_dataset = getattr(dataset_class, self.cfg.dataset)(
            self.cfg, self.p_df, valid
        )
        # 2) Initializing torch.utils.data.DataLoader Module
        loader_train = get_dataloader(self.cfg, train_dataset, self.generator)
        loader_valid = get_dataloader(self.cfg, valid_dataset, self.generator, shuffle=False, drop_last=False)
        return loader_train, loader_valid, train

    def model_setting(self, len_train: int):
        """ function for init backbone's configuration & train utils setting """
        model = getattr(model_arch, self.cfg.model_arch)(self.cfg)
        # load checkpoint when you set 'resume' to True
        if self.cfg.resume:
            model.load_state_dict(torch.load(self.cfg.checkpoint_dir + self.cfg.state_dict))
        model.to(self.cfg.device)

        criterion = getattr(model_loss, self.cfg.loss_fn)(self.cfg.reduction)
        val_criterion = getattr(model_loss, self.cfg.val_loss_fn)(self.cfg.reduction)
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
        return model, criterion, val_criterion, optimizer, lr_scheduler, awp

    def train_fn(self, loader_train, model, criterion, optimizer, scheduler, epoch, awp: None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """ function for train loop """
        # torch.autograd.set_detect_anomaly(True)
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp_scaler)
        losses, c_losses, w_losses = AverageMeter(), AverageMeter(), AverageMeter()
        model.train()
        for step, (inputs, label_content, label_wording) in enumerate(tqdm(loader_train)):
            optimizer.zero_grad()
            inputs, label_content, label_wording = one2many_collate(inputs, label_content, label_wording)

            for k, v in inputs.items():
                inputs[k] = v.to(self.cfg.device)  # train to gpu
            label_content = label_content.to(self.cfg.device)  # Two target values to GPU
            label_wording = label_wording.to(self.cfg.device)  # Two target values to GPU

            batch_size = label_content.size(0)

            with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
                pred_list = model(inputs)
                c_pred, w_pred = pred_list[:, :, 0], pred_list[:, :, 1]

                c_loss, w_loss = criterion(c_pred.view(-1, 1), label_content.view(-1, 1)), criterion(w_pred.view(-1, 1), label_wording.view(-1, 1))
                c_mask, w_mask = (label_content.view(-1, 1) != -1), (label_wording.view(-1, 1) != -1)
                c_loss, w_loss = torch.masked_select(c_loss, c_mask).mean(), torch.masked_select(w_loss, w_mask).mean()  # reduction = mean
                loss = (self.cfg.content_weight * c_loss) + (self.cfg.wording_weight * w_loss)  # Weighted MCRMSE Loss

            if self.cfg.n_gradient_accumulation_steps > 1:
                loss = loss / self.cfg.n_gradient_accumulation_steps

            scaler.scale(loss).backward()
            losses.update(loss.detach().cpu().numpy(), batch_size)  # Must do detach() for avoid memory leak
            c_losses.update(c_loss.detach().cpu().numpy(), batch_size)
            w_losses.update(w_loss.detach().cpu().numpy(), batch_size)

            # if self.cfg.awp and epoch >= self.cfg.nth_awp_start_epoch:
            #     loss = awp.attack_backward(inputs, labels)
            #     scaler.scale(loss).backward()
            #     awp._restore()

            if self.cfg.clipping_grad and (step + 1) % self.cfg.n_gradient_accumulation_steps == 0 or self.cfg.n_gradient_accumulation_steps == 1:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm(
                    model.parameters(),
                    self.cfg.max_grad_norm * self.cfg.n_gradient_accumulation_steps
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            del inputs, label_content, label_wording, pred_list, c_loss, w_loss, loss
            torch.cuda.empty_cache()
            gc.collect()

        grad_norm = grad_norm.detach().cpu().numpy()
        return losses.avg, c_losses.avg, w_losses.avg, grad_norm, scheduler.get_lr()[0]

    def valid_fn(self, loader_valid, model, val_criterion) -> Tuple[Tensor, Tensor, Tensor]:
        """ function for validation loop """
        valid_losses, c_losses, w_losses = AverageMeter(), AverageMeter(), AverageMeter()
        model.eval()
        with torch.no_grad():
            for step, (inputs, label_content, label_wording) in enumerate(tqdm(loader_valid)):
                inputs, label_content, label_wording = one2many_collate(inputs, label_content, label_wording)
                for k, v in inputs.items():
                    inputs[k] = v.to(self.cfg.device)
                label_content = label_content.to(self.cfg.device)  # Two target values to GPU
                label_wording = label_wording.to(self.cfg.device)  # Two target values to GPU
                batch_size = label_content.size(0)

                pred_list = model(inputs)
                c_pred, w_pred = pred_list[:, :, 0], pred_list[:, :, 1]

                c_loss, w_loss = val_criterion(c_pred.view(-1, 1), label_content.view(-1, 1)), val_criterion(w_pred.view(-1, 1), label_wording.view(-1, 1))
                c_mask, w_mask = (label_content.view(-1, 1) != -1), (label_wording.view(-1, 1) != -1)
                c_loss, w_loss = torch.masked_select(c_loss, c_mask).mean(), torch.masked_select(w_loss, w_mask).mean()
                loss = (c_loss + w_loss) / 2

                valid_losses.update(loss.detach().cpu().numpy(), batch_size)
                c_losses.update(c_loss.detach().cpu().numpy(), batch_size)
                w_losses.update(w_loss.detach().cpu().numpy(), batch_size)

            del inputs, label_content, label_wording, pred_list, c_loss, w_loss, loss
            torch.cuda.empty_cache()
            gc.collect()
        return valid_losses.avg, c_losses.avg, w_losses.avg


class OneToOneSmartBatchTrainer:
    """
    Trainer class for OneToOne DataSchema Pipeline, applied with Smart Batch Collate, Sampler (iterable-style)
    Smart Batch Collate & Sampler make more efficient for train/inference performance (time/memory)
    And then, rest of part are same as original OneToOneTrainer, applied many optimization techniques
    So, if you want set optimization options, go to cfg.json file or configuration.py
    Args:
        cfg: configuration module, configuration.py
        generator: torch.Generator, for init pytorch random seed
    """

    def __init__(self, cfg: CFG, generator: torch.Generator) -> None:
        self.cfg = cfg
        self.model_name = self.cfg.model.split('/')[1]
        self.generator = generator
        self.s_df = load_data('./dataset_class/data_folder/type12_summaries_train.csv')
        self.tokenizer = self.cfg.tokenizer

    def make_batch(self, fold: int) -> tuple[DataLoader, DataLoader, pd.DataFrame]:
        """ function for making batch instance for random sampler, smart-batch sampler """
        train = self.s_df[self.s_df['fold'] != fold].reset_index(drop=True)  # must have column which is already shape of prompt
        valid = self.s_df[self.s_df['fold'] == fold].reset_index(drop=True)  # must have column which is already shape of prompt

        # 1) Custom Datasets
        train_dataset = getattr(dataset_class, self.cfg.dataset)(
            self.cfg, train
        )
        valid_dataset = getattr(dataset_class, self.cfg.dataset)(
            self.cfg, valid, True
        )
        # 2) Initializing torch.utils.data.DataLoader Module
        loader_train = train_dataset.get_smart_dataloader()
        loader_valid = valid_dataset.get_smart_dataloader(drop_last=False)
        return loader_train, loader_valid, train

    def model_setting(self, len_train: int):
        """ function for init backbone's configuration & train utils setting """
        model = getattr(model_arch, self.cfg.model_arch)(self.cfg)
        # load checkpoint when you set 'resume' to True
        if self.cfg.resume:
            model.load_state_dict(torch.load(self.cfg.checkpoint_dir + self.cfg.state_dict))
        model.to(self.cfg.device)

        criterion = getattr(model_loss, self.cfg.loss_fn)(self.cfg.reduction)
        val_criterion = getattr(model_loss, self.cfg.val_loss_fn)(self.cfg.reduction)
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
        return model, criterion, val_criterion, optimizer, lr_scheduler, awp

    def train_fn(self, loader_train, model, criterion, optimizer, scheduler, epoch, awp: None) -> Tuple[Tensor, Tensor, Tensor]:
        """ function for train loop """
        # torch.autograd.set_detect_anomaly(True)
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp_scaler)
        losses = AverageMeter()
        model.train()
        for step, (inputs, labels) in enumerate(tqdm(loader_train)):
            optimizer.zero_grad()
            inputs = collate(inputs)  # need to check this method, when applying smart batching

            for k, v in inputs.items():
                inputs[k] = v.to(self.cfg.device)  # train to gpu
            labels = labels.to(self.cfg.device)  # Two target values to GPU
            batch_size = labels.size(0)

            with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
                preds = model(inputs)
                loss = criterion(preds, labels)

            if self.cfg.n_gradient_accumulation_steps > 1:
                loss = loss / self.cfg.n_gradient_accumulation_steps

            scaler.scale(loss).backward()
            losses.update(loss.detach().cpu().numpy(), batch_size)  # Must do detach() for avoid memory leak

            if self.cfg.awp and epoch >= self.cfg.nth_awp_start_epoch:
                loss = awp.attack_backward(inputs, labels)
                scaler.scale(loss).backward()
                awp._restore()

            if self.cfg.clipping_grad and (step + 1) % self.cfg.n_gradient_accumulation_steps == 0 or self.cfg.n_gradient_accumulation_steps == 1:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm(
                    model.parameters(),
                    self.cfg.max_grad_norm * self.cfg.n_gradient_accumulation_steps,
                    foreach=True
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            del inputs, labels, preds, loss
            torch.cuda.empty_cache()
            gc.collect()

        train_loss = losses.avg
        grad_norm = grad_norm.detach().cpu().numpy()
        return train_loss, grad_norm, scheduler.get_lr()[0]

    def valid_fn(self, loader_valid, model, val_criterion) -> Tensor:
        """ function for validation loop """
        valid_losses = AverageMeter()
        model.eval()
        with torch.no_grad():
            for step, (inputs, labels) in enumerate(tqdm(loader_valid)):
                inputs = collate(inputs)  # need to check this method, when applying smart batching
                for k, v in inputs.items():
                    inputs[k] = v.to(self.cfg.device)
                labels = labels.to(self.cfg.device)
                batch_size = labels.size(0)
                preds = model(inputs)
                valid_loss = val_criterion(preds, labels)
                valid_losses.update(valid_loss.detach().cpu().numpy(), batch_size)

            del inputs, labels, preds, valid_loss
            torch.cuda.empty_cache()
            gc.collect()
        valid_loss = valid_losses.avg
        return valid_loss


class OneToManyTrainer:
    """
    Trainer class for OneToMany DataSchema Pipeline, having many optimization options
    So, if you want set options, go to cfg.json file or configuration.py
    Args:
        cfg: configuration module, configuration.py
        generator: torch.Generator, for init pytorch random seed
    """
    def __init__(self, cfg: CFG, generator: torch.Generator) -> None:
        self.cfg = cfg
        self.model_name = self.cfg.model.split('/')[1]
        self.generator = generator
        self.s_df = load_data('./dataset_class/data_folder/k25_one2many_train.csv')
        self.tokenizer = self.cfg.tokenizer

    def make_batch(self, fold: int) -> tuple[DataLoader, DataLoader, pd.DataFrame]:
        """ function for making batch instance for random sampler, smart-batch sampler """
        train = self.s_df[self.s_df['fold'] != fold].reset_index(drop=True)
        valid = self.s_df[self.s_df['fold'] == fold].reset_index(drop=True)

        # 1) Custom Datasets
        train_dataset = getattr(dataset_class, self.cfg.dataset)(
            self.cfg, train
        )
        valid_dataset = getattr(dataset_class, self.cfg.dataset)(
            self.cfg, valid, is_valid=True
        )
        # 2) Initializing torch.utils.data.DataLoader Module
        loader_train = get_dataloader(self.cfg, train_dataset, self.generator)
        loader_valid = get_dataloader(self.cfg, valid_dataset, self.generator, shuffle=False, drop_last=False)
        return loader_train, loader_valid, train

    def model_setting(self, len_train: int):
        """ function for init backbone's configuration & train utils setting """
        model = getattr(model_arch, self.cfg.model_arch)(self.cfg)
        # load checkpoint when you set 'resume' to True
        if self.cfg.resume:
            model.load_state_dict(torch.load(self.cfg.checkpoint_dir + self.cfg.state_dict))
        model.to(self.cfg.device)

        criterion = getattr(model_loss, self.cfg.loss_fn)(self.cfg.reduction)
        val_criterion = getattr(model_loss, self.cfg.val_loss_fn)(self.cfg.reduction)
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
        return model, criterion, val_criterion, optimizer, lr_scheduler, awp

    def train_fn(self, loader_train, model, criterion, optimizer, scheduler, epoch, awp: None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tensor, Tensor]:
        """ function for train loop """
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp_scaler)
        losses, c_losses, w_losses = AverageMeter(), AverageMeter(), AverageMeter()
        model.train()
        for step, (inputs, _, label_content, label_wording) in enumerate(tqdm(loader_train)):
            optimizer.zero_grad()
            inputs, label_content, label_wording = one2many_collate(inputs, label_content, label_wording)  # need to check this method, when applying smart batching

            for k, v in inputs.items():
                inputs[k] = v.to(self.cfg.device)  # train to gpu
            label_content = label_content.to(self.cfg.device)  # Two target values to GPU
            label_wording = label_wording.to(self.cfg.device)  # Two target values to GPU
            batch_size = label_content.size(0)

            with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
                pred_list = model(inputs)
                c_pred, w_pred = pred_list[:, :, 0], pred_list[:, :, 1]
                c_loss, w_loss = criterion(c_pred.view(-1, 1), label_content.view(-1, 1)), criterion(w_pred.view(-1, 1), label_wording.view(-1, 1))
                c_mask, w_mask = (label_content.view(-1, 1) != -1), (label_wording.view(-1, 1) != -1)
                c_loss, w_loss = torch.masked_select(c_loss, c_mask).mean(), torch.masked_select(w_loss, w_mask).mean()  # reduction = mean
                loss = (self.cfg.content_weight * c_loss) + (self.cfg.wording_weight * w_loss)  # Weighted MCRMSE Loss

            if self.cfg.n_gradient_accumulation_steps > 1:
                loss = loss / self.cfg.n_gradient_accumulation_steps

            scaler.scale(loss).backward()
            losses.update(loss.detach().cpu().numpy(), batch_size)  # Must do detach() for avoid memory leak
            c_losses.update(c_loss.detach().cpu().numpy(), batch_size)
            w_losses.update(w_loss.detach().cpu().numpy(), batch_size)

            # if self.cfg.awp and epoch >= self.cfg.nth_awp_start_epoch:
            #     loss = awp.attack_backward(inputs, labels)
            #     scaler.scale(loss).backward()
            #     awp._restore()

            if self.cfg.clipping_grad and (step + 1) % self.cfg.n_gradient_accumulation_steps == 0 or self.cfg.n_gradient_accumulation_steps == 1:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm(
                    model.parameters(),
                    self.cfg.max_grad_norm * self.cfg.n_gradient_accumulation_steps,
                    foreach=True
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            del inputs, label_content, label_wording, loss, c_loss, w_loss
            torch.cuda.empty_cache()
            gc.collect()

        grad_norm = grad_norm.detach().cpu().numpy()
        return losses.avg, c_losses.avg, w_losses.avg, grad_norm, scheduler.get_lr()[0]

    def valid_fn(self, loader_valid, model, val_criterion) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """function for validation loop, cv metric same as cv loss, so no need to implement further more"""
        valid_losses, c_losses, w_losses = AverageMeter(), AverageMeter(), AverageMeter()
        model.eval()
        with torch.no_grad():
            for step, (inputs, _, label_content, label_wording) in enumerate(tqdm(loader_valid)):
                inputs, label_content, label_wording = one2many_collate(inputs, label_content, label_wording)  # need to check this method, when applying smart batching

                for k, v in inputs.items():
                    inputs[k] = v.to(self.cfg.device)  # train to gpu
                label_content = label_content.to(self.cfg.device)  # Two target values to GPU
                label_wording = label_wording.to(self.cfg.device)  # Two target values to GPU
                batch_size = label_content.size(0)

                pred_list = model(inputs)
                c_pred, w_pred = pred_list[:, :, 0], pred_list[:, :, 1]

                c_loss, w_loss = val_criterion(c_pred.view(-1, 1), label_content.view(-1, 1)), val_criterion(w_pred.view(-1, 1), label_wording.view(-1, 1))
                c_mask, w_mask = (label_content.view(-1, 1) != -1), (label_wording.view(-1, 1) != -1)
                c_loss, w_loss = torch.masked_select(c_loss, c_mask).mean(), torch.masked_select(w_loss, w_mask).mean()  # reduction = mean

                valid_loss = (c_loss + w_loss) / 2  # for calculating MCRMSE
                valid_losses.update(valid_loss.detach().cpu().numpy(), batch_size)
                c_losses.update(c_loss.detach().cpu().numpy(), batch_size)
                w_losses.update(w_loss.detach().cpu().numpy(), batch_size)

            del inputs, label_content, label_wording, valid_loss, c_loss, w_loss
            torch.cuda.empty_cache()
            gc.collect()

        return valid_losses.avg, c_losses.avg, w_losses.avg
