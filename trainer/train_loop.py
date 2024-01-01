import gc
import wandb
import torch
from torch.optim.swa_utils import update_bn
import numpy as np
import trainer.trainer as trainer

from tqdm.auto import tqdm
from configuration import CFG
from trainer.trainer_utils import get_name, EarlyStopping
from utils.helper import class2dict

g = torch.Generator()
g.manual_seed(CFG.seed)


def train_loop(cfg: CFG) -> None:
    """ Base Trainer Loop Function """
    wandb.init(
        project=cfg.name,
        name=f'[{cfg.arch_name}]' + cfg.module_name,
        config=class2dict(cfg),
        group=f'{cfg.module_name}/layers_{cfg.num_layers}/{cfg.mlm_masking}/max_length_{cfg.max_seq}/',
        job_type='train',
        entity="qcqced"
    )
    early_stopping = EarlyStopping(mode=cfg.stop_mode, patience=10)
    early_stopping.detecting_anomaly()

    epoch_val_score_max, val_score_max = np.inf, np.inf
    train_input = getattr(trainer, cfg.trainer)(cfg, g)  # init object
    loader_train, loader_valid, len_train = train_input.make_batch()
    model, criterion, val_criterion, val_metric_list, optimizer, lr_scheduler, awp, swa_model, swa_scheduler = train_input.model_setting(
        len_train
    )
    for epoch in tqdm(range(cfg.epochs)):
        print(f'[{epoch + 1}/{cfg.epochs}] Train & Validation')
        train_loss, val_score_max = train_input.train_val_fn(
            loader_train,
            model,
            criterion,
            optimizer,
            lr_scheduler,
            loader_valid,
            val_criterion,
            val_metric_list,
            val_score_max,
            epoch,
            awp,
            swa_model,
            cfg.swa_start,
            swa_scheduler
        )
        wandb.log({
            '<epoch> Train Loss': train_loss,
            '<epoch> Valid Loss': val_score_max,
        })
        print(f'[{epoch + 1}/{cfg.epochs}] Train Loss: {np.round(train_loss, 4)}')
        print(f'[{epoch + 1}/{cfg.epochs}] Valid Loss: {np.round(val_score_max, 4)}')

        if epoch_val_score_max >= val_score_max:
            print(f'Best Epoch Score: {val_score_max}')
            epoch_val_score_max = val_score_max
            wandb.log({
                '<epoch> Valid Loss': val_score_max,
            })
            print(f'Valid Best Loss: {np.round(val_score_max, 4)}')

        # Check if Trainer need to Early Stop
        early_stopping(val_score_max)
        if early_stopping.early_stop:
            break
        del train_loss
        gc.collect(), torch.cuda.empty_cache()

        if cfg.swa and not early_stopping.early_stop:
            update_bn(loader_train, swa_model)
            swa_loss = train_input.swa_fn(loader_valid, swa_model, val_criterion)
            print(f'[{epoch + 1}/{cfg.epochs}] SWA Val Loss: {np.round(swa_loss, 4)}')
            wandb.log({'<epoch> SWA Valid Loss': swa_loss})
            if val_score_max >= swa_loss:
                print(f'[Update] Valid Score : ({val_score_max:.4f} => {swa_loss:.4f}) Save Parameter')
                print(f'Best Score: {swa_loss}')
                torch.save(model.state_dict(),
                           f'{cfg.checkpoint_dir}_CV_{swa_loss}_{cfg.max_len}_{get_name(cfg)}_state_dict.pth')
                wandb.log({'<epoch> Valid Loss': swa_loss})

    wandb.finish()
