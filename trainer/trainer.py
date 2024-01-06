import gc
import wandb
import torch
import torch.nn as nn
import numpy as np
import transformers
from numpy import ndarray

from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel

from tqdm.auto import tqdm
from torch import Tensor
from typing import Tuple, Any, Union, List, Callable

import dataset_class.dataclass as dataset_class
from experiment.tuner import mlm, clm, sbo, rtd
from experiment.losses import loss
from experiment.metrics import metric
from model import model as task
from configuration import CFG
from dataset_class.preprocessing import load_pkl
from trainer.trainer_utils import get_optimizer_grouped_parameters, get_scheduler
from trainer.trainer_utils import AverageMeter, AWP, get_dataloader, get_swa_scheduler


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
        self.metric_list = self.cfg.metrics

    def make_batch(self) -> Tuple[DataLoader, DataLoader, int]:
        """ Function for making batch instance """
        train = load_pkl(f'./dataset_class/data_folder/{self.cfg.datafolder}/384_train')
        valid = load_pkl(f'./dataset_class/data_folder/{self.cfg.datafolder}/384_valid')

        # 1) Custom Datasets
        train_dataset = getattr(dataset_class, self.cfg.dataset)(train)
        valid_dataset = getattr(dataset_class, self.cfg.dataset)(valid)

        # 2) selecting custom masking method for each task
        collate_fn = None
        if self.cfg.task == 'MaskedLanguageModel':
            collate_fn = getattr(mlm, 'MLMCollator')(self.cfg)
        elif self.cfg.task == 'CasualLanguageModel':
            collate_fn = getattr(clm, 'CLMCollator')(self.cfg)
        elif self.cfg.task == 'SpanBoundaryObjective':
            collate_fn = getattr(sbo, 'SpanCollator')(
                self.cfg,
                self.cfg.masking_budget,
                self.cfg.span_probability,
                self.cfg.max_span_length,
            )
        elif self.cfg.task == 'ReplacedTokenDetection':
            if self.cfg.rtd_masking == 'MaskedLanguageModel':
                collate_fn = getattr(mlm, 'MLMCollator')(self.cfg)
            elif self.cfg.rtd_masking == 'SpanBoundaryObjective':
                collate_fn = getattr(sbo, 'SpanCollator')(
                    self.cfg,
                    self.cfg.masking_budget,
                    self.cfg.span_probability,
                    self.cfg.max_span_length,
                )

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
        The design is inspired by the Builder Pattern
        """
        model = getattr(task, self.cfg.task)(self.cfg)
        # load checkpoint when you set 'resume' to True
        if self.cfg.resume:
            model.load_state_dict(
                torch.load(self.cfg.checkpoint_dir + self.cfg.state_dict),
                strict=False
            )
        model.to(self.cfg.device)

        criterion = getattr(loss, self.cfg.loss_fn)(self.cfg.reduction)
        val_criterion = getattr(loss, self.cfg.val_loss_fn)(self.cfg.reduction)
        val_metric_list = [getattr(metric, f'{metrics}') for metrics in self.metric_list]
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
        return model, criterion, val_criterion, val_metric_list, optimizer, lr_scheduler, awp, swa_model, swa_scheduler

    def train_val_fn(
            self,
            loader_train,
            model: nn.Module,
            criterion: nn.Module,
            optimizer,
            scheduler,
            loader_valid,
            val_criterion: nn.Module,
            val_metric_list: List[Callable],
            val_score_max: float,
            epoch: int,
            awp: nn.Module = None,
            swa_model: nn.Module = None,
            swa_start: int = None,
            swa_scheduler=None
    ) -> Tuple[Any, Union[float, ndarray, ndarray]]:
        """ function for train loop with validation for each batch*N Steps """
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp_scaler)
        losses = AverageMeter()
        model.train()
        for step, batch in enumerate(tqdm(loader_train)):
            optimizer.zero_grad()
            inputs = batch['input_ids'].to(self.cfg.device)
            labels = batch['labels'].to(self.cfg.device)  # Two target values to GPU
            padding_mask = batch['padding_mask'].to(self.cfg.device)  # padding mask to GPU
            batch_size = inputs.size(0)
            with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
                logit = model(inputs, padding_mask)
                loss = criterion(logit.view(-1, self.cfg.vocab_size), labels.view(-1))

            if self.cfg.n_gradient_accumulation_steps > 1:
                loss = loss / self.cfg.n_gradient_accumulation_steps
            scaler.scale(loss).backward()
            losses.update(loss.detach().cpu().numpy(), batch_size)  # Must do detach() for avoid memory leak

            if self.cfg.awp and epoch >= self.cfg.nth_awp_start_epoch:
                adv_loss = awp.attack_backward(inputs, padding_mask, labels)
                scaler.scale(adv_loss).backward()
                awp._restore()

            if self.cfg.clipping_grad and (step + 1) % self.cfg.n_gradient_accumulation_steps == 0 or self.cfg.n_gradient_accumulation_steps == 1:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm(
                    model.parameters(),
                    self.cfg.max_grad_norm * self.cfg.n_gradient_accumulation_steps
                )
                # Stochastic Weight Averaging
                if self.cfg.swa and epoch >= int(swa_start):
                    swa_model.update_parameters(model)
                    swa_scheduler.step()

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            del inputs, labels, loss
            torch.cuda.empty_cache()
            gc.collect()

            # logging train loss, gradient norm, lr to wandb
            lr = scheduler.get_lr()[0]
            grad_norm = grad_norm.detach().cpu().numpy()

            wandb.log({
                '<Per Step> Train Loss': losses.avg,
                '<Per Step> Gradient Norm': grad_norm,
                '<Per Step> lr': lr,
            })

            # validate for each size of batch*N Steps
            if ((step + 1) % self.cfg.val_check == 0) or ((step + 1) == len(loader_train)):
                valid_loss, score_list = self.valid_fn(loader_valid, model, val_criterion, val_metric_list)

                print(f'[Validation Check: {step}/{len(loader_train)}] Train Loss: {np.round(losses.avg, 4)}')
                print(f'[Validation Check: {step}/{len(loader_train)}] Valid Loss: {np.round(valid_loss, 4)}')
                for i, metric_name in enumerate(self.metric_list):
                    print(f'[{step}/{len(loader_train)}] Valid {metric_name}: {score_list[i]}')
                    wandb.log({f'<Validation Check Step> Valid {metric_name}': score_list[i]})

                wandb.log({
                    '<Val Check Step> Train Loss': losses.avg,
                    '<Val Check Step> Valid Loss': valid_loss,
                })

                if val_score_max >= valid_loss:
                    print(f'[Update] Valid Score : ({val_score_max:.4f} => {valid_loss:.4f}) Save Parameter')
                    print(f'Best Score: {valid_loss}')
                    torch.save(
                        model.state_dict(),
                        f'{self.cfg.checkpoint_dir}{self.cfg.mlm_masking}_{self.cfg.max_len}_{self.cfg.module_name}_state_dict.pth'
                    )
                    val_score_max = valid_loss
                del valid_loss
                gc.collect()
                torch.cuda.empty_cache()
        return losses.avg * self.cfg.n_gradien_accumulation_steps, val_score_max

    def train_fn(self, loader_train, model, criterion, optimizer, scheduler, epoch, awp=None, swa_model=None, swa_start=None, swa_scheduler=None) -> Tuple[Tensor, Tensor, Tensor]:
        """ function for train loop
        """
        # torch.autograd.set_detect_anomaly(True)
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp_scaler)
        losses = AverageMeter()
        model.train()
        for step, batch in enumerate(tqdm(loader_train)):
            optimizer.zero_grad()
            inputs = batch['input_ids'].to(self.cfg.device)
            labels = batch['labels'].to(self.cfg.device)  # Two target values to GPU
            padding_mask = batch['padding_mask'].to(self.cfg.device)  # padding mask to GPU
            batch_size = inputs.size(0)
            with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
                logit = model(inputs, padding_mask)
                loss = criterion(logit.view(-1, self.cfg.vocab_size), labels.view(-1))

            if self.cfg.n_gradient_accumulation_steps > 1:
                loss = loss / self.cfg.n_gradient_accumulation_steps
            scaler.scale(loss).backward()
            losses.update(loss.detach().cpu().numpy(), batch_size)  # Must do detach() for avoid memory leak

            if self.cfg.awp and epoch >= self.cfg.nth_awp_start_epoch:
                adv_loss = awp.attack_backward(inputs, padding_mask, labels)
                scaler.scale(adv_loss).backward()
                awp._restore()

            if self.cfg.clipping_grad and (step + 1) % self.cfg.n_gradient_accumulation_steps == 0 or self.cfg.n_gradient_accumulation_steps == 1:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm(
                    model.parameters(),
                    self.cfg.max_grad_norm * self.cfg.n_gradient_accumulation_steps
                )
                # Stochastic Weight Averaging
                if self.cfg.swa and epoch >= int(swa_start):
                    swa_model.update_parameters(model)
                    swa_scheduler.step()

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            del inputs, labels, loss
            torch.cuda.empty_cache()
            gc.collect()

        grad_norm = grad_norm.detach().cpu().numpy()
        return losses.avg, grad_norm, scheduler.get_lr()[0]

    def valid_fn(
            self,
            loader_valid,
            model: nn.Module,
            val_criterion: nn.Module,
            val_metric_list: List[Callable]
    ) -> Tuple[np.ndarray, List]:
        """ function for validation loop
        """
        valid_losses = AverageMeter()
        valid_metrics = {self.metric_list[i]: AverageMeter() for i in range(len(self.metric_list))}
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(tqdm(loader_valid)):
                inputs = batch['input_ids'].to(self.cfg.device)
                labels = batch['labels'].to(self.cfg.device)  # Two target values to GPU
                padding_mask = batch['padding_mask'].to(self.cfg.device)  # padding mask to GPU
                batch_size = inputs.size(0)

                logit = model(inputs, padding_mask)
                flat_logit, flat_labels = logit.view(-1, self.cfg.vocab_size), labels.view(-1)

                loss = val_criterion(flat_logit, flat_labels)
                valid_losses.update(loss.detach().cpu().numpy(), batch_size)

                wandb.log({
                    '<Val Step> Valid Loss': valid_losses.avg
                })

                for i, metric_fn in enumerate(val_metric_list):
                    scores = metric_fn(flat_labels.detach().cpu().numpy(), flat_logit.detach().cpu().numpy())
                    valid_metrics[self.metric_list[i]].update(scores, batch_size)
                    wandb.log({
                        f'<Val Step> Valid {self.metric_list[i]}': scores
                    })

            del inputs, labels, loss, flat_logit, flat_labels, scores
            torch.cuda.empty_cache()
            gc.collect()
        avg_scores = [valid_metrics[self.metric_list[i]].avg for i in range(len(self.metric_list))]
        return valid_losses.avg, avg_scores

    def swa_fn(
            self,
            loader_valid,
            swa_model,
            val_criterion,
            val_metric_list: List[Callable]
    ) -> Tuple[np.ndarray, List]:
        """ Stochastic Weight Averaging, it consumes more GPU VRAM & training times """
        swa_model.eval()
        valid_losses = AverageMeter()
        valid_metrics = {self.metric_list[i]: AverageMeter() for i in range(len(self.metric_list))}
        with torch.no_grad():
            for step, batch in enumerate(tqdm(loader_valid)):
                inputs = batch['input_ids'].to(self.cfg.device)
                labels = batch['labels'].to(self.cfg.device)  # Two target values to GPU
                padding_mask = batch['padding_mask'].to(self.cfg.device)  # padding mask to GPU
                batch_size = inputs.size(0)

                logit = swa_model(inputs, padding_mask)
                flat_logit, flat_labels = logit.view(-1, self.cfg.vocab_size), labels.view(-1)

                loss = val_criterion(flat_logit, flat_labels)
                valid_losses.update(loss.detach().cpu().numpy(), batch_size)
                for i, metric_fn in enumerate(val_metric_list):
                    scores = metric_fn(flat_labels.detach().cpu().numpy(), flat_logit.detach().cpu().numpy())
                    valid_metrics[self.metric_list[i]].update(scores, batch_size)

            del inputs, labels, loss, flat_logit, flat_labels, scores
            torch.cuda.empty_cache()
            gc.collect()
        avg_scores = [valid_metrics[self.metric_list[i]].avg for i in range(len(self.metric_list))]
        return valid_losses.avg, avg_scores


class SBOTuner(PreTrainTuner):
    """ Trainer class for Span Boundary Objective
    """
    def __init__(self, cfg: CFG, generator: torch.Generator) -> None:
        super(SBOTuner, self).__init__(cfg, generator)

    def train_val_fn(
            self,
            loader_train,
            model: nn.Module,
            criterion: nn.Module,
            optimizer,
            scheduler,
            loader_valid,
            val_criterion: nn.Module,
            val_metric_list: List[Callable],
            val_score_max: float,
            epoch: int,
            awp: nn.Module = None,
            swa_model: nn.Module = None,
            swa_start: int = None,
            swa_scheduler=None
    ) -> Tuple[Any, Union[float, ndarray, ndarray]]:
        """ function for train loop with validation for each batch*N Steps
        SpanBERT has two loss, one is MLM loss, the other is SBO loss
        """
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp_scaler)
        losses, mlm_losses, sbo_losses = AverageMeter(), AverageMeter(), AverageMeter()
        model.train()
        for step, batch in enumerate(tqdm(loader_train)):
            optimizer.zero_grad()
            inputs = batch['input_ids'].to(self.cfg.device)
            labels = batch['labels'].to(self.cfg.device)  # Two target values to GPU
            padding_mask = batch['padding_mask'].to(self.cfg.device)  # padding mask to GPU
            mask_labels = batch['mask_labels'].to(self.cfg.device)  # mask labels to GPU

            batch_size = inputs.size(0)
            with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
                mlm_logit, sbo_logit = model(
                    inputs=inputs,
                    padding_mask=padding_mask,
                    mask_labels=mask_labels
                )
                mlm_loss = criterion(mlm_logit.view(-1, self.cfg.vocab_size), labels.view(-1))
                sbo_loss = criterion(sbo_logit.view(-1, self.cfg.vocab_size), labels.view(-1))
                loss = mlm_loss + sbo_loss

            if self.cfg.n_gradient_accumulation_steps > 1:
                loss = loss / self.cfg.n_gradient_accumulation_steps

            scaler.scale(loss).backward()
            losses.update(loss.detach().cpu().numpy(), batch_size)  # Must do detach() for avoid memory leak
            mlm_losses.update(mlm_loss.detach().cpu().numpy(), batch_size)
            sbo_losses.update(sbo_loss.detach().cpu().numpy(), batch_size)

            if self.cfg.awp and epoch >= self.cfg.nth_awp_start_epoch:  # later update
                adv_loss = awp.attack_backward(inputs, padding_mask, labels)
                scaler.scale(adv_loss).backward()
                awp._restore()

            if self.cfg.clipping_grad and (step + 1) % self.cfg.n_gradient_accumulation_steps == 0 or self.cfg.n_gradient_accumulation_steps == 1:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm(
                    model.parameters(),
                    self.cfg.max_grad_norm * self.cfg.n_gradient_accumulation_steps
                )
                # Stochastic Weight Averaging
                if self.cfg.swa and epoch >= int(swa_start):
                    swa_model.update_parameters(model)
                    swa_scheduler.step()

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            del inputs, labels, loss, mlm_loss, sbo_loss, mlm_logit, sbo_logit, mask_labels, padding_mask
            torch.cuda.empty_cache()
            gc.collect()

            # logging train loss, gradient norm, lr to wandb
            lr = scheduler.get_lr()[0]
            grad_norm = grad_norm.detach().cpu().numpy()

            wandb.log({
                '<Per Step> Total Train Loss': losses.avg,
                '<Per Step> MLM Train Loss': mlm_losses.avg,
                '<Per Step> SBO Train Loss': sbo_losses.avg,
                '<Per Step> Gradient Norm': grad_norm,
                '<Per Step> lr': lr,
            })

            # validate for each size of batch*N Steps
            if ((step + 1) % self.cfg.val_check == 0) or ((step + 1) == len(loader_train)):
                valid_loss, mlm_valid_loss, sbo_valid_loss, mlm_score_list, sbo_score_list = self.valid_fn(
                    loader_valid,
                    model,
                    val_criterion,
                    val_metric_list
                )
                print(f'[Validation Check: {step}/{len(loader_train)}] Total Train Loss: {np.round(losses.avg, 4)}')
                print(f'[Validation Check: {step}/{len(loader_train)}] MLM Train Loss: {np.round(mlm_losses.avg, 4)}')
                print(f'[Validation Check: {step}/{len(loader_train)}] SBO Train Loss: {np.round(sbo_losses.avg, 4)}')

                print(f'[Validation Check: {step}/{len(loader_train)}] Total Valid Loss: {np.round(valid_loss, 4)}')
                print(f'[Validation Check: {step}/{len(loader_train)}] MLM Valid Loss: {np.round(mlm_valid_loss, 4)}')
                print(f'[Validation Check: {step}/{len(loader_train)}] SBO Valid Loss: {np.round(sbo_valid_loss, 4)}')

                for i, metric_name in enumerate(self.metric_list):
                    print(f'[{step}/{len(loader_train)}] MLM Valid {metric_name}: {mlm_score_list[i]}')
                    print(f'[{step}/{len(loader_train)}] SBO Valid {metric_name}: {sbo_score_list[i]}')
                    wandb.log({
                        f'<Validation Check Step> MLM Valid {metric_name}': mlm_score_list[i],
                        f'<Validation Check Step> SBO Valid {metric_name}': sbo_score_list[i]
                    })

                wandb.log({
                    '<Val Check Step> Total Train Loss': losses.avg,
                    '<Val Check Step> MLM Train Loss': mlm_losses.avg,
                    '<Val Check Step> SBO Train Loss': sbo_losses.avg,
                    '<Val Check Step> Total Valid Loss': valid_loss,
                    '<Val Check Step> MLM Valid Loss': mlm_valid_loss,
                    '<Val Check Step> SBO Valid Loss': sbo_valid_loss,
                })

                if val_score_max >= valid_loss:
                    print(f'[Update] Valid Score : ({val_score_max:.4f} => {valid_loss:.4f}) Save Parameter')
                    print(f'Best Score: {valid_loss}')
                    torch.save(
                        model.state_dict(),
                        f'{self.cfg.checkpoint_dir}{self.cfg.task}_SpanBERT_{self.cfg.max_len}_{self.cfg.module_name}_state_dict.pth'
                    )
                    val_score_max = valid_loss
                del valid_loss
                gc.collect()
                torch.cuda.empty_cache()
        return losses.avg * self.cfg.n_gradien_accumulation_steps, val_score_max

    def valid_fn(
            self,
            loader_valid,
            model: nn.Module,
            val_criterion: nn.Module,
            val_metric_list: List[Callable]
    ) -> Tuple[Any, Any, Any, List[Any], List[Any]]:
        """ function for validation loop
        """
        valid_losses, valid_mlm_losses, valid_sbo_losses = AverageMeter(), AverageMeter(), AverageMeter()
        mlm_valid_metrics = {self.metric_list[i]: AverageMeter() for i in range(len(self.metric_list))}
        sbo_valid_metrics = {self.metric_list[i]: AverageMeter() for i in range(len(self.metric_list))}
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(tqdm(loader_valid)):
                inputs = batch['input_ids'].to(self.cfg.device)
                labels = batch['labels'].to(self.cfg.device)  # Two target values to GPU
                padding_mask = batch['padding_mask'].to(self.cfg.device)  # padding mask to GPU
                mask_labels = batch['mask_labels'].to(self.cfg.device)  # mask labels to GPU
                batch_size = inputs.size(0)

                mlm_logit, sbo_logit = model(
                    inputs=inputs,
                    padding_mask=padding_mask,
                    mask_labels=mask_labels
                )
                labels = labels.view(-1)
                mlm_logit, sbo_logit = mlm_logit.view(-1, self.cfg.vocab_size), sbo_logit.view(-1, self.cfg.vocab_size)

                mlm_loss = val_criterion(mlm_logit, labels)
                sbo_loss = val_criterion(sbo_logit, labels)
                loss = mlm_loss + sbo_loss

                valid_losses.update(loss.detach().cpu().numpy(), batch_size)
                valid_mlm_losses.update(mlm_loss.detach().cpu().numpy(), batch_size)
                valid_sbo_losses.update(sbo_loss.detach().cpu().numpy(), batch_size)

                wandb.log({
                    '<Val Step> Total Valid Loss': valid_losses.avg,
                    '<Val Step> MLM Valid Loss': valid_mlm_losses.avg,
                    '<Val Step> SBO Valid Loss': valid_sbo_losses.avg,
                })

                for i, metric_fn in enumerate(val_metric_list):
                    mlm_scores = metric_fn(
                        mlm_logit.detach().cpu().numpy(),
                        labels.detach().cpu().numpy()
                    )
                    sbo_scores = metric_fn(
                        sbo_logit.detach().cpu().numpy(),
                        labels.detach().cpu().numpy()
                    )
                    mlm_valid_metrics[self.metric_list[i]].update(mlm_scores, batch_size)
                    sbo_valid_metrics[self.metric_list[i]].update(sbo_scores, batch_size)
                    wandb.log({
                        f'<Val Step> MLM Valid {self.metric_list[i]}': mlm_scores,
                        f'<Val Step> SBO Valid {self.metric_list[i]}': sbo_scores,
                    })

            del inputs, labels, loss, mlm_loss, sbo_loss, mlm_logit, sbo_logit, padding_mask, mask_labels
            torch.cuda.empty_cache()
            gc.collect()
        mlm_avg_scores = [mlm_valid_metrics[self.metric_list[i]].avg for i in range(len(self.metric_list))]
        sbo_avg_scores = [sbo_valid_metrics[self.metric_list[i]].avg for i in range(len(self.metric_list))]
        return valid_losses.avg, valid_mlm_losses.avg, valid_sbo_losses.avg, mlm_avg_scores, sbo_avg_scores


class RTDTuner(PreTrainTuner):
    """ Trainer class for Replaced Token Detection
    """
    def __init__(self, cfg: CFG, generator: torch.Generator) -> None:
        super(RTDTuner, self).__init__(cfg, generator)
        self.model_name = self.cfg.module_name
        self.generator_name = self.cfg.generator
        self.discriminator_name = self.cfg.discriminator

    def model_setting(self, len_train: int):
        """ Function for init backbone's configuration & train utils setting,
        The design is inspired by the Builder Pattern
        """
        model = getattr(task, self.cfg.task)(self.cfg)
        # load checkpoint when you set 'resume' to True
        if self.cfg.is_generator_resume:
            model.model.generator.load_state_dict(
                torch.load(self.cfg.checkpoint_dir + self.cfg.state_dict)
            )
        if self.cfg.is_discriminator_resume:
            model.model.discriminator.load_state_dict(
                torch.load(self.cfg.checkpoint_dir + self.cfg.state_dict)
            )
        model.to(self.cfg.device)

        criterion = getattr(loss, self.cfg.loss_fn)(self.cfg.reduction)
        val_criterion = getattr(loss, self.cfg.val_loss_fn)(self.cfg.reduction)
        val_metric_list = [getattr(metric, f'{metrics}') for metrics in self.metric_list]
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
        return model, criterion, val_criterion, val_metric_list, optimizer, lr_scheduler, awp, swa_model, swa_scheduler

    def train_val_fn(
            self,
            loader_train,
            model: nn.Module,
            criterion: nn.Module,
            optimizer,
            scheduler,
            loader_valid,
            val_criterion: nn.Module,
            val_metric_list: List[Callable],
            val_score_max: float,
            epoch: int,
            awp: nn.Module = None,
            swa_model: nn.Module = None,
            swa_start: int = None,
            swa_scheduler=None
    ) -> Tuple[Any, Union[float, ndarray, ndarray]]:
        """ function for train loop with validation for each batch*N Steps
        ELECTRA has two loss, one is generator loss, the other is discriminator loss Each of two losses are quite different,
        Models can be underfitted like tag-of-war if they simply sum losses with different characteristics
        in situations where they share word embeddings, or backwards as it were.

        This is a demo version, so it's a simple matrix sum and backwards, but in the future we'll develop several gradient update methods
        like GDES as described in the DeBERTa-V3 paper.
        """
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp_scaler)
        losses, g_losses, d_losses = AverageMeter(), AverageMeter(), AverageMeter()
        model.train()
        for step, batch in enumerate(tqdm(loader_train)):
            optimizer.zero_grad()
            inputs = batch['input_ids'].to(self.cfg.device)
            labels = batch['labels'].to(self.cfg.device)  # Two target values to GPU
            padding_mask = batch['padding_mask'].to(self.cfg.device)  # padding mask to GPU
            batch_size = inputs.size(0)
            with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
                g_logit, d_logit, d_labels = model(inputs, labels, padding_mask)  # generator logit, discriminator logit
                g_loss = criterion(g_logit.view(-1, self.cfg.vocab_size), labels.view(-1))
                d_loss = criterion(d_logit.view(-1, 2), d_labels)
                loss = g_loss + d_loss

            if self.cfg.n_gradient_accumulation_steps > 1:
                loss = loss / self.cfg.n_gradient_accumulation_steps

            scaler.scale(loss).backward()
            losses.update(loss.detach().cpu().numpy(), batch_size)  # Must do detach() for avoid memory leak
            g_losses.update(g_loss.detach().cpu().numpy(), batch_size)
            d_losses.update(d_loss.detach().cpu().numpy(), batch_size)

            if self.cfg.awp and epoch >= self.cfg.nth_awp_start_epoch:  # later update
                adv_loss = awp.attack_backward(inputs, padding_mask, labels)
                scaler.scale(adv_loss).backward()
                awp._restore()

            if self.cfg.clipping_grad and (step + 1) % self.cfg.n_gradient_accumulation_steps == 0 or self.cfg.n_gradient_accumulation_steps == 1:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm(
                    model.parameters(),
                    self.cfg.max_grad_norm * self.cfg.n_gradient_accumulation_steps
                )
                # Stochastic Weight Averaging
                if self.cfg.swa and epoch >= int(swa_start):
                    swa_model.update_parameters(model)
                    swa_scheduler.step()

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            del inputs, labels, loss, g_loss, d_loss, g_logit, d_logit, d_labels
            torch.cuda.empty_cache()
            gc.collect()

            # logging train loss, gradient norm, lr to wandb
            lr = scheduler.get_lr()[0]
            grad_norm = grad_norm.detach().cpu().numpy()

            wandb.log({
                '<Per Step> Total Train Loss': losses.avg,
                '<Per Step> Generator Train Loss': g_losses.avg,
                '<Per Step> Discriminator Train Loss': d_losses.avg,
                '<Per Step> Gradient Norm': grad_norm,
                '<Per Step> lr': lr,
            })

            # validate for each size of batch*N Steps
            if ((step + 1) % self.cfg.val_check == 0) or ((step + 1) == len(loader_train)):
                valid_loss, g_valid_loss, d_valid_loss, g_score_list, d_score_list = self.valid_fn(
                    loader_valid,
                    model,
                    val_criterion,
                    val_metric_list
                )
                print(f'[Validation Check: {step}/{len(loader_train)}] Total Train Loss: {np.round(losses.avg, 4)}')
                print(f'[Validation Check: {step}/{len(loader_train)}] Generator Train Loss: {np.round(g_losses.avg, 4)}')
                print(f'[Validation Check: {step}/{len(loader_train)}] Discriminator Train Loss: {np.round(d_losses.avg, 4)}')

                print(f'[Validation Check: {step}/{len(loader_train)}] Total Valid Loss: {np.round(valid_loss, 4)}')
                print(f'[Validation Check: {step}/{len(loader_train)}] Generator Valid Loss: {np.round(g_valid_loss, 4)}')
                print(f'[Validation Check: {step}/{len(loader_train)}] Discriminator Valid Loss: {np.round(d_valid_loss, 4)}')

                for i, metric_name in enumerate(self.metric_list):
                    print(f'[{step}/{len(loader_train)}] Generator Valid {metric_name}: {g_score_list[i]}')
                    print(f'[{step}/{len(loader_train)}] Discriminator Valid {metric_name}: {d_score_list[i]}')
                    wandb.log({
                        f'<Validation Check Step> Generator Valid {metric_name}': g_score_list[i],
                        f'<Validation Check Step> Discriminator Valid {metric_name}': d_score_list[i]
                    })

                wandb.log({
                    '<Val Check Step> Total Train Loss': losses.avg,
                    '<Val Check Step> Generator Train Loss': g_losses.avg,
                    '<Val Check Step> Discriminator Train Loss': d_losses.avg,
                    '<Val Check Step> Total Valid Loss': valid_loss,
                    '<Val Check Step> Generator Valid Loss': g_valid_loss,
                    '<Val Check Step> Discriminator Valid Loss': d_valid_loss,
                })

                if val_score_max >= valid_loss:
                    print(f'[Update] Valid Score : ({val_score_max:.4f} => {valid_loss:.4f}) Save Parameter')
                    print(f'Best Score: {valid_loss}')
                    torch.save(
                        model.state_dict(),
                        f'{self.cfg.checkpoint_dir}{self.cfg.rtd_masking}_{self.cfg.mlm_masking}_ELECTRA_{self.cfg.max_len}_{self.cfg.module_name}_state_dict.pth'
                    )
                    val_score_max = valid_loss
                del valid_loss
                gc.collect()
                torch.cuda.empty_cache()
        return losses.avg * self.cfg.n_gradien_accumulation_steps, val_score_max

    def valid_fn(
            self,
            loader_valid,
            model: nn.Module,
            val_criterion: nn.Module,
            val_metric_list: List[Callable]
    ) -> Tuple[Any, Any, Any, List[Any], List[Any]]:
        """ function for validation loop
        """
        valid_losses, valid_g_losses, valid_d_losses = AverageMeter(), AverageMeter(), AverageMeter()
        g_valid_metrics = {self.metric_list[i]: AverageMeter() for i in range(len(self.metric_list))}
        d_valid_metrics = {self.metric_list[i]: AverageMeter() for i in range(len(self.metric_list))}
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(tqdm(loader_valid)):
                inputs = batch['input_ids'].to(self.cfg.device)
                labels = batch['labels'].to(self.cfg.device)  # Two target values to GPU
                padding_mask = batch['padding_mask'].to(self.cfg.device)  # padding mask to GPU
                batch_size = inputs.size(0)

                g_logit, d_logit, d_labels = model(inputs, padding_mask)
                g_loss = val_criterion(g_logit.view(-1, self.cfg.vocab_size), labels.view(-1))
                d_loss = val_criterion(d_logit.view(-1, self.cfg.vocab_size), d_labels)
                loss = g_loss + d_loss

                valid_losses.update(loss.detach().cpu().numpy(), batch_size)
                valid_g_losses.update(g_loss.detach().cpu().numpy(), batch_size)
                valid_d_losses.update(d_loss.detach().cpu().numpy(), batch_size)

                wandb.log({
                    '<Val Step> Valid Loss': valid_losses.avg,
                    '<Val Step> Generator Valid Loss': valid_g_losses.avg,
                    '<Val Step> Discriminator Valid Loss': valid_d_losses.avg,
                })

                for i, metric_fn in enumerate(val_metric_list):
                    g_scores = metric_fn(
                        g_logit.view(-1, self.cfg.vocab_size).detach().cpu().numpy(),
                        labels.view(-1).detach().cpu().numpy()
                    )
                    d_scores = metric_fn(
                        d_logit.view(-1, self.cfg.vocab_size).detach().cpu().numpy(),
                        d_labels.detach().cpu().numpy()
                    )
                    g_valid_metrics[self.metric_list[i]].update(g_scores, batch_size)
                    d_valid_metrics[self.metric_list[i]].update(d_scores, batch_size)
                    wandb.log({
                        f'<Val Step> Generator Valid {self.metric_list[i]}': g_scores,
                        f'<Val Step> Discriminator Valid {self.metric_list[i]}': d_scores,
                    })

            del inputs, labels, loss, g_loss, d_loss, g_scores, d_scores, g_logit, d_logit, d_labels
            torch.cuda.empty_cache()
            gc.collect()
        g_avg_scores = [g_valid_metrics[self.metric_list[i]].avg for i in range(len(self.metric_list))]
        d_avg_scores = [d_valid_metrics[self.metric_list[i]].avg for i in range(len(self.metric_list))]
        return valid_losses.avg, valid_g_losses.avg, valid_d_losses.avg, g_avg_scores, d_avg_scores

