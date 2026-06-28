from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from garl_ttc.config import write_yaml
from garl_ttc.datasets import TTCEstimationDataset
from garl_ttc.engine.runtime import load_checkpoint, pick_device
from garl_ttc.models import TTCNetwork


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_optimizer(model: torch.nn.Module, cfg: dict):
    params = [p for p in model.parameters() if p.requires_grad]
    optim_cfg = cfg['optimizer']
    if optim_cfg['optim_type'] == 'adam':
        optimizer = torch.optim.Adam(params, lr=optim_cfg['lr'], weight_decay=optim_cfg['weight_decay'])
    elif optim_cfg['optim_type'] == 'sgd':
        optimizer = torch.optim.SGD(
            params,
            lr=optim_cfg['lr'],
            momentum=optim_cfg['momentum'],
            weight_decay=optim_cfg['weight_decay'],
        )
    else:
        raise NotImplementedError(f"Unsupported optimizer: {optim_cfg['optim_type']}")
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=optim_cfg['milestones'],
        gamma=optim_cfg['gamma'],
    )
    return optimizer, scheduler


def make_train_loader(dataset: TTCEstimationDataset, cfg: dict) -> DataLoader:
    settings = cfg['training_settings']
    return DataLoader(
        dataset,
        batch_size=settings['batch_size'],
        num_workers=settings.get('num_threads', 0),
        shuffle=settings.get('shuffle', True),
        collate_fn=dataset.get_collate_fn(),
        pin_memory=settings.get('pin_memory', False),
    )


def _state_dict(model: torch.nn.Module) -> dict:
    return model.module.state_dict() if hasattr(model, 'module') else model.state_dict()


def train(
    cfg: dict,
    *,
    output_dir: str | Path | None = None,
    device: str | None = None,
    epochs: int | None = None,
    max_batches: int | None = None,
    seed: int = 0,
) -> Path:
    seed_everything(seed)
    torch.backends.cudnn.benchmark = cfg['cudnn']['benchmark']
    torch.backends.cudnn.deterministic = cfg['cudnn']['deterministic']
    torch.backends.cudnn.enabled = cfg['cudnn']['enabled']

    torch_device = pick_device(device)
    out_root = Path(output_dir or cfg['dirs']['output'])
    run_dir = out_root / cfg['exp_type'] / time.strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(cfg, run_dir / 'config.yaml')

    dataset = TTCEstimationDataset(cfg, 'train', seed=seed, db_mode='window')
    loader = make_train_loader(dataset, cfg)

    model = TTCNetwork(cfg, is_train=True).to(torch_device)
    if cfg['training_settings'].get('resume'):
        load_checkpoint(model, cfg['training_settings']['ckpt_path'], torch_device, strict=True)

    gpu_ids = cfg.get('gpu_id', []) if torch_device.type == 'cuda' else []
    if torch_device.type == 'cuda' and len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    optimizer, scheduler = prepare_optimizer(model, cfg)
    total_epochs = epochs or cfg['training_settings']['total_epochs']
    snapshot_epochs = set(cfg['training_settings'].get('snapshot_epochs', []))
    report_every = cfg['training_settings'].get('report_every', 20)
    for epoch in range(1, total_epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f'Epoch {epoch}', ncols=100)
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue
            data = batch['data'].to(torch_device, non_blocking=True)
            target = batch['target'].to(torch_device, non_blocking=True)
            visible_height = batch.get('visible_height')
            mask_target = batch.get('mask_target')
            if visible_height is not None:
                visible_height = visible_height.to(torch_device, non_blocking=True)
            if mask_target is not None:
                mask_target = mask_target.to(torch_device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            net = model.module if hasattr(model, 'module') else model
            _, _, loss_dict, print_dict = net.forward_train(
                data,
                target,
                visible_height_target=visible_height,
                mask_target=mask_target,
                use_mask_supervison=batch.get('use_mask_supervison'),
                epoch_idx=epoch,
            )
            loss_total = sum(loss_dict.values())
            loss_total.backward()
            optimizer.step()

            if batch_idx % report_every == 0:
                loss_msg = ' '.join(f'{k}={v.item():.4f}' for k, v in loss_dict.items())
                metric_msg = ' '.join(f'{k}={v.item():.4f}' for k, v in print_dict.items())
                pbar.set_postfix_str(f'loss={loss_total.item():.4f} {loss_msg} {metric_msg}')
            if max_batches is not None and batch_idx + 1 >= max_batches:
                break

        scheduler.step()
        if epoch in snapshot_epochs:
            ckpt_path = run_dir / f"{cfg['exp_type']}_{epoch}.pth"
            torch.save(_state_dict(model), ckpt_path)
            print(f'Saved checkpoint: {ckpt_path}')

    final_path = run_dir / 'ckpt.pth'
    torch.save(_state_dict(model), final_path)
    print(f'Saved final checkpoint: {final_path}')
    return final_path
