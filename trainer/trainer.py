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
from typing import Tuple, Any, Union, List, Callable, Dict

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
    """ Trainer class for Pre-Train Pipeline, such as MLM, CLM
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
        if self.cfg.task in ['MaskedLanguageModel', 'DistillationKnowledge']:
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

        optimizer = getattr(transformers, self.cfg.optimizer)(
            params=model.parameters(),
            lr=self.cfg.lr,
            betas=self.cfg.betas,
            eps=self.cfg.adam_epsilon,
            weight_decay=self.cfg.weight_decay,
            correct_bias=not self.cfg.use_bertadam
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
            val_score_max_2: float,
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
        return losses.avg * self.cfg.n_gradient_accumulation_steps, val_score_max

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
        avg_scores = [valid_metrics[self.metric_list[i]].avg for i in range(len(self.metric_list))]
        return valid_losses.avg, avg_scores


class SBOTuner(PreTrainTuner):
    """ Trainer class for Span Boundary Objective Task
    you can select any other implemented Auto-Encoder Model to backbone of this task,
    please check now available models in README.md
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
            val_score_max_2: float,
            epoch: int,
            awp: nn.Module = None,
            swa_model: nn.Module = None,
            swa_start: int = None,
            swa_scheduler=None
    ) -> tuple[Any, Union[float, Any]]:
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
        return losses.avg * self.cfg.n_gradient_accumulation_steps, val_score_max

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
        mlm_avg_scores = [mlm_valid_metrics[self.metric_list[i]].avg for i in range(len(self.metric_list))]
        sbo_avg_scores = [sbo_valid_metrics[self.metric_list[i]].avg for i in range(len(self.metric_list))]
        return valid_losses.avg, valid_mlm_losses.avg, valid_sbo_losses.avg, mlm_avg_scores, sbo_avg_scores


class RTDTuner(PreTrainTuner):
    """ Trainer class for Replaced Token Detection, you can use three types of training options

    1) Embedding Sharing (ES) (Implemented):
        Generator & Discriminator share word embedding, Backward jointly with sum of two losses
        (loss = generator loss + discriminator loss)

    2) No Embedding Sharing (NES) (Not Implemented):
        Generator & Discriminator do not share same word embedding, Backward separately with two losses
        generator loss update ONLY generator's weight, discriminator loss update ONLY discriminator's weight
        This method prevents 'tag-of-war' problem, but has pain point: it takes more time & memory to train

    3) Gradient Disentangled Embedding Sharing (GDES) (Implemented):
        Generator & Discriminator share word embedding limited, GDES Algorithms are described blow:
            1) share generator & discriminator's word embedding
            2) calculate MLM loss, backward with MLM loss for updating generator's word embedding (shared)
            3) make inputs for discriminator used by generator's output
            4) initialize Delta Embedding matrix with zero matrix, and then
               sum Delta E, Generator's E(must be detached), called Discriminator's E
            5) calculate BCE loss, backward with BCE loss for updating Discriminator's E

    we return only discriminator's train & validation loss, because we use discriminator's checkpoint for fine-tuning
    but save both generator & discriminator's state_dict for tuning process
    """
    def __init__(self, cfg: CFG, generator: torch.Generator) -> None:
        super(RTDTuner, self).__init__(cfg, generator)
        self.model_name = self.cfg.module_name
        self.share_embed_method = self.cfg.share_embed_method

    def model_setting(self, len_train: int):
        """ Function for init backbone's configuration & train utils setting,
        The design is inspired by the Builder Pattern
        """
        model = getattr(task, self.cfg.task)(self.cfg)

        # load checkpoint when you set 'resume' to True
        if self.cfg.is_generator_resume:
            model.model.generator.load_state_dict(
                torch.load(self.cfg.checkpoint_dir + self.cfg.generator_state_dict),
                strict=False
            )
        if self.cfg.is_discriminator_resume:
            model.model.discriminator.load_state_dict(
                torch.load(self.cfg.checkpoint_dir + self.cfg.discriminator_state_dict),
                strict=True
            )
        model.to(self.cfg.device)

        criterion = getattr(loss, self.cfg.loss_fn)(self.cfg.reduction)  # make list of loss function same as val metric logicp
        val_criterion = getattr(loss, self.cfg.val_loss_fn)(self.cfg.reduction)
        val_metric_list = [getattr(metric, f'{metrics}') for metrics in self.metric_list]

        optimizer = getattr(transformers, self.cfg.optimizer)(
            params=model.parameters(),
            lr=self.cfg.lr,
            betas=self.cfg.betas,
            eps=self.cfg.adam_epsilon,
            weight_decay=self.cfg.weight_decay,
            correct_bias=not self.cfg.use_bertadam
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
            g_val_score_max: float,
            d_val_score_max: float,
            epoch: int,
            awp: nn.Module = None,
            swa_model: nn.Module = None,
            swa_start: int = None,
            swa_scheduler=None
    ) -> Tuple[Any, Union[float, Any]]:
        """ Function for train loop with validation for each batch*N Steps
        ELECTRA has two loss, one is generator loss, the other is discriminator loss Each of two losses are quite different,
        Models can be under-fitted like tag-of-war if they simply sum losses with different characteristics
        in situations where they share word embeddings, or backwards as it were.

        This Method is implemented for Embedding Sharing
        Discriminator's loss can't backward to Generator, except for Embedding Layers (word, position)
        Embedding Layers are updated by sum of two losses
        """
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp_scaler)
        losses, g_losses, d_losses = AverageMeter(), AverageMeter(), AverageMeter()
        model.train()
        for step, batch in enumerate(tqdm(loader_train)):
            optimizer.zero_grad()
            inputs = batch['input_ids'].to(self.cfg.device)
            labels = batch['labels'].to(self.cfg.device)  # Two target values to GPU
            padding_mask = batch['padding_mask'].to(self.cfg.device)  # padding mask to GPU
            batch_size = inputs.size(0)  # same as ES, GDES

            with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
                g_logit, d_inputs, d_labels = model.generator_fw(
                    inputs,
                    labels,
                    padding_mask
                )
                d_logit = model.discriminator_fw(
                    d_inputs,
                    padding_mask
                )
                g_loss = criterion(g_logit.view(-1, self.cfg.vocab_size), labels.view(-1))
                d_loss = criterion(d_logit.view(-1, 2), d_labels)
                loss = g_loss + self.cfg.discriminator_lambda*d_loss  # discriminator's loss can't backward to generator

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
            """
            1) validate for each size of batch*N Steps
            2) save each part of model's checkpoint when BEST validation score is updated
            """
            if ((step + 1) % self.cfg.val_check == 0) or ((step + 1) == len(loader_train)):
                g_valid_loss, d_valid_loss, g_score_list, d_score_list = self.valid_fn(
                    loader_valid,
                    model,
                    val_criterion,
                    val_metric_list
                )
                valid_loss = g_valid_loss + d_valid_loss
                print(f'[Validation Check: {step}/{len(loader_train)}] Total Train Loss: {np.round(losses.avg, 4)}')
                print(
                    f'[Validation Check: {step}/{len(loader_train)}] Generator Train Loss: {np.round(g_losses.avg, 4)}')
                print(
                    f'[Validation Check: {step}/{len(loader_train)}] Discriminator Train Loss: {np.round(d_losses.avg, 4)}')

                print(f'[Validation Check: {step}/{len(loader_train)}] Total Valid Loss: {np.round(valid_loss, 4)}')
                print(
                    f'[Validation Check: {step}/{len(loader_train)}] Generator Valid Loss: {np.round(g_valid_loss, 4)}')
                print(
                    f'[Validation Check: {step}/{len(loader_train)}] Discriminator Valid Loss: {np.round(d_valid_loss, 4)}')

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

                # save checkpoint of generator
                if g_val_score_max >= g_valid_loss:
                    print(f'[Update] Generator Valid Score : ({g_val_score_max:.4f} => {g_valid_loss:.4f}) Save Parameter')
                    print(f'Generator Best Score: {g_valid_loss}')
                    torch.save(
                        model.model.generator.state_dict(),
                        f'{self.cfg.checkpoint_dir}ELECTRA_Generator_{self.cfg.rtd_masking}_{self.cfg.mlm_masking}{self.cfg.max_len}_{self.cfg.module_name}_state_dict.pth'
                    )
                    g_val_score_max = g_valid_loss

                # save checkpoint of Discriminator
                if d_val_score_max >= d_valid_loss:
                    print(f'[Update] Discriminator Valid Score : ({d_val_score_max:.4f} => {d_valid_loss:.4f}) Save Parameter')
                    print(f'Discriminator Best Score: {d_valid_loss}')
                    torch.save(
                        model.model.discriminator.state_dict(),
                        f'{self.cfg.checkpoint_dir}ELECTRA_Discriminator_{self.cfg.rtd_masking}_{self.cfg.mlm_masking}{self.cfg.max_len}_{self.cfg.module_name}_state_dict.pth'
                    )
                    d_val_score_max = d_valid_loss
        return d_losses.avg * self.cfg.n_gradient_accumulation_steps, d_val_score_max

    def gdes_train_val_fn(
            self,
            loader_train,
            model: nn.Module,
            criterion: nn.Module,
            optimizer,
            scheduler,
            loader_valid,
            val_criterion: nn.Module,
            val_metric_list: List[Callable],
            g_val_score_max: float,
            d_val_score_max: float,
            epoch: int = None,
            awp: nn.Module = None,
    ) -> Tuple[Any, Union[float, Any]]:
        """ Function for train loop with validation for each batch*N Steps
        ELECTRA has two loss, one is generator loss, the other is discriminator loss Each of two losses are quite different,
        Models can be under-fitted like tag-of-war if they simply sum losses with different characteristics
        in situations where they share word embeddings, or backwards as it were.

        This Method is implemented for Gradient Disentangled Embedding Sharing (GDES)
        Discriminator's loss can't backward to Generator, also not updating Embedding Layers (word, position) too
        Generator's Embedding Layers are updated by ONLY Generator's Loss
        """
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp_scaler)
        g_losses, d_losses = AverageMeter(), AverageMeter()
        model.train()
        for step, batch in enumerate(tqdm(loader_train)):
            optimizer.zero_grad()
            inputs = batch['input_ids'].to(self.cfg.device)
            labels = batch['labels'].to(self.cfg.device)  # Two target values to GPU
            padding_mask = batch['padding_mask'].to(self.cfg.device)  # padding mask to GPU
            batch_size = inputs.size(0)

            # Tune for Generator
            with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
                g_logit, d_inputs, d_labels = model.generator_fw(
                    inputs,
                    labels,
                    padding_mask
                )
                g_loss = criterion(g_logit.view(-1, self.cfg.vocab_size), labels.view(-1))
            if self.cfg.n_gradient_accumulation_steps > 1:
                g_loss = g_loss / self.cfg.n_gradient_accumulation_steps

            scaler.scale(g_loss).backward()
            g_losses.update(g_loss.detach().cpu().numpy(), batch_size)

            if self.cfg.clipping_grad and (step + 1) % self.cfg.n_gradient_accumulation_steps == 0 or self.cfg.n_gradient_accumulation_steps == 1:
                scaler.unscale_(optimizer)
                g_grad_norm = torch.nn.utils.clip_grad_norm(
                    model.parameters(),
                    self.cfg.max_grad_norm * self.cfg.n_gradient_accumulation_steps
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            # Tune for Discriminator
            with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
                d_logit = model.discriminator_fw(
                    d_inputs,
                    padding_mask
                )
                d_loss = criterion(d_logit.view(-1, 2), d_labels) * self.cfg.discriminator_lambda  # lambda is hyper-parameter, in original paper, lambda is 50

            if self.cfg.n_gradient_accumulation_steps > 1:
                d_losses = d_losses / self.cfg.n_gradient_accumulation_steps

            scaler.scale(d_loss).backward()
            d_losses.update(d_loss.detach().cpu().numpy(), batch_size)

            if self.cfg.clipping_grad and (step + 1) % self.cfg.n_gradient_accumulation_steps == 0 or self.cfg.n_gradient_accumulation_steps == 1:
                scaler.unscale_(optimizer)
                d_grad_norm = torch.nn.utils.clip_grad_norm(
                    model.parameters(),
                    self.cfg.max_grad_norm * self.cfg.n_gradient_accumulation_steps
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            # logging train loss, gradient norm, lr to wandb
            lr = scheduler.get_lr()[0]
            g_grad_norm = g_grad_norm.detach().cpu().numpy()
            d_grad_norm = d_grad_norm.detach().cpu().numpy()

            avg_loss = g_losses.avg + d_losses.avg
            wandb.log({
                '<Per Step> Total Train Loss': avg_loss,
                '<Per Step> Generator Train Loss': g_losses.avg,
                '<Per Step> Discriminator Train Loss': d_losses.avg,
                '<Per Step> Generator Gradient Norm': g_grad_norm,
                '<Per Step> Discriminator Gradient Norm': d_grad_norm,
                '<Per Step> lr': lr,
            })

            """ 1) validate for each size of batch*N Steps
            2) save each part of model's checkpoint when BEST validation score is updated
            """
            if ((step + 1) % self.cfg.val_check == 0) or ((step + 1) == len(loader_train)):
                g_valid_loss, d_valid_loss, g_score_list, d_score_list = self.valid_fn(
                    loader_valid,
                    model,
                    val_criterion,
                    val_metric_list
                )
                valid_loss = g_valid_loss + d_valid_loss
                print(f'[Validation Check: {step}/{len(loader_train)}] Total Train Loss: {np.round(avg_loss, 4)}')
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
                    '<Val Check Step> Total Train Loss': avg_loss,
                    '<Val Check Step> Generator Train Loss': g_losses.avg,
                    '<Val Check Step> Discriminator Train Loss': d_losses.avg,
                    '<Val Check Step> Total Valid Loss': valid_loss,
                    '<Val Check Step> Generator Valid Loss': g_valid_loss,
                    '<Val Check Step> Discriminator Valid Loss': d_valid_loss,
                })

                # save checkpoint of generator
                if g_val_score_max >= g_valid_loss:
                    print(
                        f'[Update] Generator Valid Score : ({g_val_score_max:.4f} => {g_valid_loss:.4f}) Save Parameter')
                    print(f'Generator Best Score: {g_valid_loss}')
                    torch.save(
                        model.model.generator.state_dict(),
                        f'{self.cfg.checkpoint_dir}ELECTRA_Generator_{self.cfg.rtd_masking}_{self.cfg.mlm_masking}{self.cfg.max_len}_{self.cfg.module_name}_state_dict.pth'
                    )
                    g_val_score_max = g_valid_loss

                # save checkpoint of Discriminator
                if d_val_score_max >= d_valid_loss:
                    print(f'[Update] Discriminator Valid Score : ({d_val_score_max:.4f} => {d_valid_loss:.4f}) Save Parameter')
                    print(f'Discriminator Best Score: {d_valid_loss}')
                    torch.save(
                        model.model.discriminator.state_dict(),
                        f'{self.cfg.checkpoint_dir}ELECTRA_Discriminator_{self.cfg.rtd_masking}_{self.cfg.mlm_masking}{self.cfg.max_len}_{self.cfg.module_name}_state_dict.pth'
                    )
                    d_val_score_max = d_valid_loss
        return d_losses.avg * self.cfg.n_gradient_accumulation_steps, d_val_score_max

    def valid_fn(
        self,
        loader_valid,
        model: nn.Module,
        val_criterion: nn.Module,
        val_metric_list: List[Callable]
    ) -> Tuple[Any, Any, List[Any], List[Any]]:
        """ method for pure Embedding Sharing, Gradient-Disentangled Embedding Sharing ELECTRA validation loop
        """
        valid_g_losses, valid_d_losses = AverageMeter(), AverageMeter()
        g_valid_metrics = {self.metric_list[i]: AverageMeter() for i in range(len(self.metric_list))}
        d_valid_metrics = {self.metric_list[i]: AverageMeter() for i in range(len(self.metric_list))}
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(tqdm(loader_valid)):
                inputs = batch['input_ids'].to(self.cfg.device)
                labels = batch['labels'].to(self.cfg.device)
                padding_mask = batch['padding_mask'].to(self.cfg.device)
                batch_size = inputs.size(0)

                # Generator validation part
                g_logit, d_inputs, d_labels = model.generator_fw(
                    inputs,
                    labels,
                    padding_mask
                )
                g_loss = val_criterion(g_logit.view(-1, self.cfg.vocab_size), labels.view(-1))

                # Discriminator validation part
                d_logit = model.discriminator_fw(
                    d_inputs,
                    padding_mask
                )
                d_loss = val_criterion(d_logit.view(-1, 2), d_labels)

                valid_g_losses.update(g_loss.detach().cpu().numpy(), batch_size)
                valid_d_losses.update(d_loss.detach().cpu().numpy(), batch_size)

                valid_loss = valid_g_losses.avg + valid_d_losses.avg
                wandb.log({
                    '<Val Step> Valid Loss': valid_loss,
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
        g_avg_scores = [g_valid_metrics[self.metric_list[i]].avg for i in range(len(self.metric_list))]
        d_avg_scores = [d_valid_metrics[self.metric_list[i]].avg for i in range(len(self.metric_list))]
        return valid_g_losses.avg, valid_d_losses.avg, g_avg_scores, d_avg_scores


class DistillKnowledgeTuner(PreTrainTuner):
    """ Trainer class for Distill Knowledge Pipeline, you can use three types of training options
    Now, This module only support for Distill Knowledge with Auto-Encoder Based Model like as BERT, DeBERTa ... etc
    ASAP, we will update this module to support different architecture like as RNN, LSTM, GRU ... etc

    Args:
        cfg: configuration module, configuration.py
        generator: torch.Generator, for init pytorch random seed
    """
    def __init__(self, cfg: CFG, generator: torch.Generator) -> None:
        super(DistillKnowledgeTuner, self).__init__(cfg, generator)

    def model_setting(self, len_train: int):
        """ Function for init backbone's configuration & train utils setting,
        The design is inspired by the Builder Pattern
        """
        model = getattr(task, self.cfg.task)(self.cfg)

        # load checkpoint when you set 'resume' to True
        if self.cfg.is_teacher_resume:  # load teacher's pretrained weight: backbone & mlm head
            tmp_teacher = model.teacher + model.mlm_head
            tmp_teacher.load_state_dict(
                torch.load(self.cfg.checkpoint_dir + self.cfg.teacher_state_dict),
                strict=True
            )
        if self.cfg.is_student_resume:  # load student's checkpoint
            model.student.load_state_dict(
                torch.load(self.cfg.checkpoint_dir + self.cfg.student_state_dict),
                strict=False
            )

        model.to(self.cfg.device)

        criterion = {loss_fn: getattr(loss, f'{loss_fn}') for loss_fn in self.cfg.losses_fn}
        val_criterion = {val_loss_fn: getattr(loss, f'{val_loss_fn}') for val_loss_fn in self.cfg.val_losses_fn}
        val_metric_list = [getattr(metric, f'{metrics}') for metrics in self.metric_list]

        optimizer = getattr(transformers, self.cfg.optimizer)(
            params=model.parameters(),
            lr=self.cfg.lr,
            betas=self.cfg.betas,
            eps=self.cfg.adam_epsilon,
            weight_decay=self.cfg.weight_decay,
            correct_bias=not self.cfg.use_bertadam
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
            criterion: Dict[str, nn.Module],
            optimizer,
            scheduler,
            loader_valid,
            val_criterion: Dict[str, nn.Module],
            val_metric_list: List[Callable],
            val_score_max: float,
            val_score_max_2: float,
            epoch: int,
            awp: nn.Module = None,
            swa_model: nn.Module = None,
            swa_start: int = None,
            swa_scheduler=None
    ) -> Tuple[Any, Union[float, Any]]:
        """ Function for train loop with validation for each batch*N Steps
        DistillBERT has three loss:
            1) distillation loss, calculated by soft targets & soft predictions
                (nn.KLDIVLoss(reduction='batchmean'))
            2) student loss, calculated by hard targets & hard predictions
                (nn.CrossEntropyLoss(reduction='mean')), same as pure MLM Loss
            3) cosine similarity loss, calculated by student & teacher logit similarity
                (nn.CosineEmbeddingLoss(reduction='mean')), similar as contrastive loss
        Those 3 losses are summed jointly and then backward to student model
        """
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp_scaler)
        d_losses, s_losses, c_losses = AverageMeter(), AverageMeter(), AverageMeter()
        model.train()
        for step, batch in enumerate(tqdm(loader_train)):
            optimizer.zero_grad()
            inputs = batch['input_ids'].to(self.cfg.device)
            labels = batch['labels'].to(self.cfg.device)  # Two target values to GPU
            padding_mask = batch['padding_mask'].to(self.cfg.device)  # padding mask to GPU
            batch_size = inputs.size(0)  # same as ES, GDES

            with torch.no_grad():
                t_hidden_state, soft_target = model.teacher_fw(
                    inputs,
                    padding_mask
                )  # teacher model's pred => hard logit

            with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
                s_hidden_state, s_logit, soft_pred = model.student_fw(
                    inputs,
                    labels,
                    padding_mask
                )
                d_loss = criterion["KLDivLoss"](soft_pred, soft_target)  # nn.KLDIVLoss
                s_loss = criterion["CrossEntropyLoss"](s_logit.view(-1, self.cfg.vocab_size), labels.view(-1))  # nn.CrossEntropyLoss
                c_loss = criterion["CosineEmbeddingLoss"](s_hidden_state, t_hidden_state)  # nn.CosineEmbeddingLoss
                loss = d_loss + s_loss + c_loss  # linear combination loss

            if self.cfg.n_gradient_accumulation_steps > 1:
                loss = loss / self.cfg.n_gradient_accumulation_steps

            scaler.scale(loss).backward()

            d_losses.update(d_loss.detach().cpu().numpy(), batch_size)
            s_losses.update(s_loss.detach().cpu().numpy(), batch_size)  # Must do detach() for avoid memory leak
            c_losses.update(c_loss.detach().cpu().numpy(), batch_size)

            if self.cfg.awp and epoch >= self.cfg.nth_awp_start_epoch:  # later update, current version is not supported
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

            # logging train loss, gradient norm, lr to wandb
            lr = scheduler.get_lr()[0]
            grad_norm = grad_norm.detach().cpu().numpy()

            avg_loss = d_losses.avg + s_losses.avg + c_losses.avg
            wandb.log({
                '<Per Step> Total Train Loss': avg_loss,
                '<Per Step> Distillation Train Loss': d_losses.avg,
                '<Per Step> Student Train Loss': s_losses.avg,
                '<Per Step> Cosine Embedding Loss Train Loss': c_losses.avg,
                '<Per Step> Gradient Norm': grad_norm,
                '<Per Step> lr': lr,
            })
            """
            1) validate for each size of batch*N Steps
            2) save each part of model's checkpoint when BEST validation score is updated
            """
            if ((step + 1) % self.cfg.val_check == 0) or ((step + 1) == len(loader_train)):
                d_valid_loss, s_valid_loss, c_valid_loss = self.valid_fn(
                    loader_valid,
                    model,
                    val_criterion,
                    val_metric_list
                )
                valid_loss = d_valid_loss + s_valid_loss + c_valid_loss

                # save checkpoint of ONLY student, not including mlm head
                if val_score_max >= valid_loss:
                    print(f'[Update] Total Valid Score : ({val_score_max:.4f} => {valid_loss:.4f}) Save Parameter')
                    print(f'Total Best Score: {valid_loss}')
                    torch.save(
                        model.model.student.state_dict(),
                        f'{self.cfg.checkpoint_dir}DistilBERT_Student_{self.cfg.mlm_masking}_{self.cfg.max_len}_{self.cfg.module_name}_state_dict.pth'
                    )
                    val_score_max = valid_loss

        return d_losses.avg * self.cfg.n_gradient_accumulation_steps, val_score_max

    def valid_fn(
        self,
        loader_valid,
        model: nn.Module,
        val_criterion: Dict[str, nn.Module],
        val_metric_list: List[Callable]
    ) -> Tuple[float, float, float]:
        """ method for validating DistillBERT model, this model recover temperature for pure softmax(T=1)
        """
        valid_d_losses, valid_s_losses, valid_c_losses = AverageMeter(), AverageMeter(), AverageMeter()
        valid_metrics = {self.metric_list[i]: AverageMeter() for i in range(len(self.metric_list))}
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(tqdm(loader_valid)):
                inputs = batch['input_ids'].to(self.cfg.device)
                labels = batch['labels'].to(self.cfg.device)
                padding_mask = batch['padding_mask'].to(self.cfg.device)
                batch_size = inputs.size(0)

                # 1) Teacher model valid pred
                t_hidden_state, soft_target = model.teacher_fw(
                    inputs,
                    padding_mask,
                    is_valid=True
                )

                # 2) Student model valid pred
                s_hidden_state, s_logit, soft_pred = model.student_fw(
                    inputs,
                    labels,
                    padding_mask
                )
                d_loss = val_criterion["KLDivLoss"](soft_pred, soft_target)  # nn.KLDIVLoss
                s_loss = val_criterion["CrossEntropyLoss"](s_logit.view(-1, self.cfg.vocab_size), labels.view(-1))  # nn.CrossEntropyLoss
                c_loss = val_criterion["CosineEmbeddingLoss"](s_hidden_state, t_hidden_state)  # nn.CosineEmbeddingLoss

                valid_d_losses.update(d_loss.detach().cpu().numpy(), batch_size)
                valid_s_losses.update(s_loss.detach().cpu().numpy(), batch_size)
                valid_c_losses.update(c_loss.detach().cpu().numpy(), batch_size)

                valid_loss = valid_d_losses.avg + valid_s_losses.avg + valid_c_losses.avg
                wandb.log({
                    '<Val Step> Valid Loss': valid_loss,
                    '<Val Step> Distillation Valid Loss': valid_d_losses.avg,
                    '<Val Step> Student Valid Loss': valid_s_losses.avg,
                    '<Val Step> Cosine Embedding Valid Loss': valid_c_losses.avg,
                })

                for i, metric_fn in enumerate(val_metric_list):
                    if self.metric_list[i] == 'accuracy':
                        scores = metric_fn(
                            s_logit.view(-1, self.cfg.vocab_size).detach().cpu().numpy(),
                            labels.view(-1).detach().cpu().numpy()
                        )
                    elif self.metric_list[i] == 'cosine_similarity':
                        scores = metric_fn(
                            s_hidden_state.detach().cpu().numpy(),
                            t_hidden_state.detach().cpu().numpy()
                        )
                    valid_metrics[self.metric_list[i]].update(scores, batch_size)
                    wandb.log({
                        f'<Val Step> Student Valid {self.metric_list[i]}': scores,
                    })
        return valid_d_losses.avg, valid_s_losses.avg, valid_c_losses.avg


class FineTuningTuner:
    """ Trainer class for Fine-Tuning Pipeline, such as SQUAD, Glue, SuperGlue ... etc
    This module will be updated to support various tasks ASAP

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
