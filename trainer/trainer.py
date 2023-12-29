import gc
import wandb
import torch
import numpy as np
import transformers
from numpy import ndarray

from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel

from tqdm.auto import tqdm
from torch import Tensor
from typing import Tuple, Any, Union, List, Callable

import dataset_class.dataclass as dataset_class
from experiment.tuner import mlm, clm
from experiment.losses import loss
from experiment.metrics import metric
from model import model as task
from configuration import CFG
from dataset_class.preprocessing import load_pkl
from trainer.trainer_utils import get_optimizer_grouped_parameters, get_scheduler, get_name
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
        model = getattr(task, self.cfg.task)(self.cfg)
        # load checkpoint when you set 'resume' to True
        if self.cfg.resume:
            model.load_state_dict(torch.load(self.cfg.checkpoint_dir + self.cfg.state_dict))
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
            model,
            criterion,
            optimizer,
            scheduler,
            loader_valid,
            val_criterion,
            val_metric_list: List[Callable],
            val_score_max: float,
            epoch: int,
            awp=None,
            swa_model=None,
            swa_start=None,
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

                print(f'[{step}/{len(loader_train)}] Train Loss: {np.round(losses.avg, 4)}')
                print(f'[{step}/{len(loader_train)}] Valid Loss: {np.round(valid_loss, 4)}')
                for i, metric_name in enumerate(self.metric_list):
                    print(f'[{step}/{len(loader_train)}] Valid {metric_name}: {score_list[i]}')
                    wandb.log({f'<Step> Valid {metric_name}': score_list[i]})

                wandb.log({
                    '<Train & Valid> Train Loss': losses.avg,
                    '<Train & Valid> Valid Loss': valid_loss,
                })

                if val_score_max >= valid_loss:
                    print(f'[Update] Valid Score : ({val_score_max:.4f} => {valid_loss:.4f}) Save Parameter')
                    print(f'Best Score: {valid_loss}')
                    torch.save(
                        model.state_dict(),
                        f'{self.cfg.checkpoint_dir}_{self.cfg.max_len}_{self.cfg.module_name}_state_dict.pth'
                    )
                    val_score_max = valid_loss
                del valid_loss
                gc.collect()
                torch.cuda.empty_cache()
        return losses.avg * self.cfg.n_gradient_accumulation_steps, val_score_max

    def train_fn(self, loader_train, model, criterion, optimizer, scheduler, epoch, awp=None, swa_model=None, swa_start=None, swa_scheduler=None) -> Tuple[Tensor, Tensor, Tensor]:
        """ function for train loop """
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

    def valid_fn(self, loader_valid, model, val_criterion, val_metric_list: List[Callable]) -> Tuple[np.ndarray, List]:
        """ function for validation loop """
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
                loss = val_criterion(logit.view(-1, self.cfg.vocab_size), labels.view(-1))
                valid_losses.update(loss.detach().cpu().numpy(), batch_size)
                for i, metric_fn in enumerate(val_metric_list):
                    scores = metric_fn(logit.detach().cpu().numpy(), labels.detach().cpu().numpy())
                    valid_metrics[self.metric_list[i]].update(scores, batch_size)

            del inputs, labels, loss, scores
            torch.cuda.empty_cache()
            gc.collect()
        avg_scores = [valid_metrics[self.metric_list[i]].avg for i in range(len(self.metric_list))]
        return valid_losses.avg, avg_scores

    def swa_fn(self, loader_valid, swa_model, val_criterion, val_metric_list: List[Callable]) -> Tuple[np.ndarray, List]:
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
                loss = val_criterion(logit.view(-1, self.cfg.vocab_size), labels.view(-1))
                valid_losses.update(loss.detach().cpu().numpy(), batch_size)
                for i, metric_fn in enumerate(val_metric_list):
                    scores = metric_fn(logit.detach().cpu().numpy(), labels.detach().cpu().numpy())
                    valid_metrics[self.metric_list[i]].update(scores, batch_size)

            del inputs, labels, loss, scores
            torch.cuda.empty_cache()
            gc.collect()
        avg_scores = [valid_metrics[self.metric_list[i]].avg for i in range(len(self.metric_list))]
        return valid_losses.avg, avg_scores
