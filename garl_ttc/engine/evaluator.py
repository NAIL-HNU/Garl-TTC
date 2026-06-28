from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from garl_ttc.datasets import TTCEstimationDataset
from garl_ttc.engine.metrics import collect_predictions, print_summary, summarize_results
from garl_ttc.engine.runtime import load_checkpoint, pick_device
from garl_ttc.models import TTCNetwork


def make_test_loader(dataset: TTCEstimationDataset, cfg: dict) -> DataLoader:
    settings = cfg['testing_settings']
    return DataLoader(
        dataset,
        batch_size=settings.get('batch_size', 1),
        num_workers=settings.get('num_threads', 0),
        shuffle=settings.get('shuffle', False),
        collate_fn=dataset.get_collate_fn(),
        pin_memory=settings.get('pin_memory', False),
    )


def evaluate(
    cfg: dict,
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    device: str | None = None,
    max_batches: int | None = None,
    db_mode: str = 'sequence',
    seq_ts: int = 5,
    test_ttc_range: tuple[float, float] = (-10, 10),
    strict: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    torch_device = pick_device(device)
    model = TTCNetwork(cfg, is_train=False).to(torch_device)
    load_checkpoint(model, checkpoint, torch_device, strict=strict)
    model.eval()

    dataset = TTCEstimationDataset(
        cfg,
        'test',
        seed=0,
        db_mode=db_mode,
        seq_ts=seq_ts,
        test_ttc_range=list(test_ttc_range),
    )
    loader = make_test_loader(dataset, cfg)

    rows = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc='Eval', ncols=100)):
            if batch is None:
                continue
            data = batch['data'].to(torch_device, non_blocking=True)
            target = batch['target'].to(torch_device, non_blocking=True)
            visible_height = batch.get('visible_height')
            if visible_height is not None:
                visible_height = visible_height.to(torch_device, non_blocking=True)
            prediction, _ = model(data)
            rows.extend(collect_predictions(
                prediction,
                target,
                visible_height=visible_height,
                case_id=batch.get('case_id'),
                seq_name=batch.get('seq_name'),
                pred_mode=cfg['model']['mode'],
                dT=model.dT,
            ))
            if max_batches is not None and batch_idx + 1 >= max_batches:
                break

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = out_dir / 'eval_results_sorted.csv'
    df.to_csv(csv_path, index=False)

    summary = summarize_results(df) if not df.empty else pd.DataFrame()
    if not summary.empty:
        summary.to_csv(out_dir / 'summary.csv', index=False)
        print_summary(summary)
    print(f'Saved predictions to {csv_path}')
    return df, summary
