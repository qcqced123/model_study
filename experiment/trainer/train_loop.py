import gc
import wandb
import torch
from torch.optim.swa_utils import update_bn

import numpy as np
from tqdm.auto import tqdm
import trainer.trainer as trainer
from configuration import CFG
from trainer.trainer_utils import get_name, EarlyStopping
from utils.helper import class2dict
g = torch.Generator()
g.manual_seed(CFG.seed)


def train_loop(cfg: CFG) -> None:
    """ Base Trainer Loop Function """
    fold_list = [i for i in range(cfg.n_folds)]
    for fold in tqdm(fold_list):
        print(f'============== {fold}th Fold Train & Validation ==============')
        wandb.init(
            project=cfg.name,
            name=f'[{cfg.model_arch}]' + cfg.model + f'/fold{fold}',
            config=class2dict(cfg),
            group=f'(Wording)/{cfg.n_folds}/prompt2/{cfg.loss_fn}/{cfg.model}/max_length_{cfg.max_len}/',
            job_type='train',
            entity="qcqced"
        )
        early_stopping = EarlyStopping(mode=cfg.stop_mode, patience=10)
        early_stopping.detecting_anomaly()

        val_score_max = np.inf
        train_input = getattr(trainer, cfg.name)(cfg, g)  # init object
        loader_train, loader_valid, train = train_input.make_batch(fold)
        model, criterion, val_criterion, optimizer, lr_scheduler, awp, swa_model, swa_scheduler = train_input.model_setting(len(train))

        for epoch in range(cfg.epochs):
            print(f'[{epoch + 1}/{cfg.epochs}] Train & Validation')
            train_loss, c_loss, w_loss, grad_norm, lr = train_input.train_fn(
                loader_train, model, criterion, optimizer, lr_scheduler, epoch, awp,
                swa_model, cfg.swa_start, swa_scheduler
            )
            valid_loss, v_c_loss, v_w_loss = train_input.valid_fn(
                loader_valid, model, val_criterion
            )
            wandb.log({
                '<epoch> Train Loss': train_loss,
                '<epoch> Train Content Loss': c_loss,
                '<epoch> Train Wording Loss': w_loss,
                '<epoch> Valid Loss': valid_loss,
                '<epoch> Valid Content Loss': v_c_loss,
                '<epoch> Valid Wording Loss': v_w_loss,
                '<epoch> Gradient Norm': grad_norm,
                '<epoch> lr': lr
            })
            print(f'[{epoch + 1}/{cfg.epochs}] Train Loss: {np.round(train_loss, 4)}')
            print(f'[{epoch + 1}/{cfg.epochs}] Train Content Loss: {np.round(c_loss, 4)}')
            print(f'[{epoch + 1}/{cfg.epochs}] Train Wording Loss: {np.round(w_loss, 4)}')
            print(f'[{epoch + 1}/{cfg.epochs}] Valid Loss: {np.round(valid_loss, 4)}')
            print(f'[{epoch + 1}/{cfg.epochs}] Valid Content Loss: {np.round(v_c_loss, 4)}')
            print(f'[{epoch + 1}/{cfg.epochs}] Valid Wording Loss: {np.round(v_w_loss, 4)}')
            print(f'[{epoch + 1}/{cfg.epochs}] Gradient Norm: {np.round(grad_norm, 4)}')
            print(f'[{epoch + 1}/{cfg.epochs}] lr: {lr}')

            if val_score_max >= valid_loss:
                print(f'[Update] Valid Score : ({val_score_max:.4f} => {valid_loss:.4f}) Save Parameter')
                print(f'Best Score: {valid_loss}')
                torch.save(
                    model.state_dict(),
                    f'{cfg.checkpoint_dir}fold{fold}_CV_{valid_loss}_{cfg.pooling}_{cfg.max_len}_{get_name(cfg)}_state_dict.pth'
                )
                val_score_max = valid_loss

            # Check if Trainer need to Early Stop
            early_stopping(valid_loss)
            if early_stopping.early_stop:
                break
            del train_loss, valid_loss, grad_norm, lr
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
                               f'{cfg.checkpoint_dir}_SWA_fold{fold}_CV_{swa_loss}_{cfg.pooling}_{cfg.max_len}_{get_name(cfg)}_state_dict.pth')
                    wandb.log({'<epoch> Valid Loss': swa_loss})

        wandb.finish()
