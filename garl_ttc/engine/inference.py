from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from garl_ttc.datasets import TTCEstimationDataset
from garl_ttc.engine.runtime import load_checkpoint, pick_device
from garl_ttc.models import TTCNetwork


def make_inference_loader(dataset: TTCEstimationDataset, cfg: dict) -> DataLoader:
    settings = cfg['testing_settings']
    return DataLoader(
        dataset,
        batch_size=settings.get('batch_size', 1),
        num_workers=settings.get('num_threads', 0),
        shuffle=False,
        collate_fn=dataset.get_collate_fn(),
        pin_memory=settings.get('pin_memory', False),
    )


def prediction_to_ttc(prediction: torch.Tensor, *, pred_mode: str, dT: float) -> np.ndarray:
    pred = prediction.detach().cpu().numpy()
    n = len(pred)
    if pred_mode == 'height_ratio':
        visual_heights = pred.reshape(n, -1)
        pred_height_ratio = visual_heights[:, 0] / visual_heights[:, 1]
        return dT / (1.0 - pred_height_ratio)
    if pred_mode == 'height_ratio_direct':
        pred_height_ratio = pred.reshape(n, -1)[:, 0]
        return dT / (1.0 - pred_height_ratio)
    if pred_mode == 'baseline':
        return pred.reshape(n, -1)[:, 0]
    raise NotImplementedError(f'Unsupported pred_mode: {pred_mode}')


def run_inference(
    cfg: dict,
    checkpoint: str | Path,
    output_json: str | Path,
    *,
    device: str | None = None,
    max_batches: int | None = None,
    seq_ts: int = 5,
    strict: bool = True,
) -> dict:
    torch_device = pick_device(device)
    model = TTCNetwork(cfg, is_train=False).to(torch_device)
    load_checkpoint(model, checkpoint, torch_device, strict=strict)
    model.eval()

    dataset = TTCEstimationDataset(
        cfg,
        'test',
        seed=0,
        db_mode='sequence',
        seq_ts=seq_ts,
        test_ttc_range=(-10, 10),
    )
    loader = make_inference_loader(dataset, cfg)

    results = {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc='Infer', ncols=100)):
            if batch is None:
                continue
            data = batch['data'].to(torch_device, non_blocking=True)
            prediction, _ = model(data)
            pred_ttc = prediction_to_ttc(prediction, pred_mode=cfg['model']['mode'], dT=model.dT)
            tokens = batch.get('sample_token') or []
            if len(tokens) != len(pred_ttc):
                raise RuntimeError(f'prediction/token length mismatch: {len(pred_ttc)} vs {len(tokens)}')
            for token, ttc in zip(tokens, pred_ttc):
                results[str(token)] = {'ttc': float(ttc)}
            if max_batches is not None and batch_idx + 1 >= max_batches:
                break

    payload = {
        'meta': {
            'format': 'garlttc_prediction_v1',
        },
        'results': results,
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + '\n')
    print(f'Saved GarlTTC submission JSON to {output_json}')
    return payload
