#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from garl_ttc.config import load_config, set_config_value
from garl_ttc.datasets import TTCEstimationDataset
from garl_ttc.engine.evaluator import make_test_loader
from garl_ttc.engine.metrics import collect_predictions, print_summary, summarize_results
from garl_ttc.engine.runtime import load_checkpoint, pick_device
from garl_ttc.models import TTCNetwork


VARIANTS = {
    'visual_baseline': {
        'label': 'Visual-only baseline',
        'config': 'configs/ablation/visual_baseline.yaml',
        'checkpoint': 'checkpoints/paper_visual_only_baseline.pth',
        'input': 'rgb',
    },
    'event_baseline': {
        'label': 'Event-only baseline',
        'config': 'configs/ablation/event_baseline.yaml',
        'checkpoint': 'checkpoints/paper_event_only_baseline.pth',
        'input': 'event',
    },
    'multimodal_baseline': {
        'label': 'Multi-modal baseline',
        'config': 'configs/ablation/multimodal_baseline.yaml',
        'checkpoint': 'checkpoints/paper_multi_modal_baseline.pth',
        'input': 'full',
    },
    'visual_lhr': {
        'label': 'Visual-only + LHR',
        'config': 'configs/ablation/visual_lhr.yaml',
        'checkpoint': 'checkpoints/paper_visual_only_lhr.pth',
        'input': 'rgb',
    },
    'event_lhr': {
        'label': 'Event-only + LHR',
        'config': 'configs/ablation/event_lhr.yaml',
        'checkpoint': 'checkpoints/paper_event_only_lhr.pth',
        'input': 'event',
    },
    'multimodal_lhr_early': {
        'label': 'Multi-modal + LHR early',
        'config': 'configs/ablation/multimodal_lhr_early.yaml',
        'checkpoint': 'checkpoints/paper_multi_modal_lhr_early_fusion.pth',
        'input': 'full',
    },
    'multimodal_lhr_late': {
        'label': 'Multi-modal + LHR late',
        'config': 'configs/ablation/multimodal_lhr_late.yaml',
        'checkpoint': 'checkpoints/paper_multi_modal_lhr_late_fusion.pth',
        'input': 'full',
    },
    'ours_full': {
        'label': 'Ours (Full)',
        'config': 'configs/ablation/ours_full.yaml',
        'checkpoint': 'checkpoints/paper_ours_full.pth',
        'input': 'full',
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate paper ablation variants with shared benchmark preprocessing.')
    parser.add_argument('--variants', default=','.join(VARIANTS), help='Comma-separated variant keys, or all.')
    parser.add_argument('--output-dir', default=str(ROOT / 'outputs/ablation_full'))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-batches', type=int, default=None)
    parser.add_argument('--seq-ts', type=int, default=5)
    parser.add_argument('--ttc-min', type=float, default=-10)
    parser.add_argument('--ttc-max', type=float, default=10)
    return parser.parse_args()


def select_input(data: torch.Tensor, input_kind: str) -> torch.Tensor:
    if input_kind == 'rgb':
        return data[:, :6]
    if input_kind == 'event':
        return data[:, 6:]
    if input_kind == 'full':
        return data
    raise ValueError(f'Unsupported input kind: {input_kind}')


def main() -> None:
    args = parse_args()
    if args.variants == 'all':
        variant_keys = list(VARIANTS)
    else:
        variant_keys = [key.strip() for key in args.variants.split(',') if key.strip()]
    unknown = [key for key in variant_keys if key not in VARIANTS]
    if unknown:
        raise ValueError(f'Unknown variants: {unknown}')

    device = pick_device(args.device)
    dataset_cfg = load_config(VARIANTS['multimodal_baseline']['config'])
    set_config_value(dataset_cfg, 'dataset.mode', 'image_event')
    set_config_value(dataset_cfg, 'testing_settings.batch_size', args.batch_size)
    set_config_value(dataset_cfg, 'testing_settings.num_threads', args.num_workers)

    models = {}
    for key in variant_keys:
        spec = VARIANTS[key]
        cfg = load_config(spec['config'])
        model = TTCNetwork(cfg, is_train=False).to(device)
        load_checkpoint(model, spec['checkpoint'], device)
        model.eval()
        models[key] = {'spec': spec, 'cfg': cfg, 'model': model, 'rows': []}

    dataset = TTCEstimationDataset(
        dataset_cfg,
        'test',
        seed=0,
        db_mode='sequence',
        seq_ts=args.seq_ts,
        test_ttc_range=[args.ttc_min, args.ttc_max],
    )
    loader = make_test_loader(dataset, dataset_cfg)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc='Ablation Eval', ncols=100)):
            if batch is None:
                continue
            data = batch['data'].to(device, non_blocking=True)
            target = batch['target'].to(device, non_blocking=True)
            visible_height = batch.get('visible_height')
            if visible_height is not None:
                visible_height = visible_height.to(device, non_blocking=True)

            for key, item in models.items():
                spec = item['spec']
                cfg = item['cfg']
                model = item['model']
                prediction, _ = model(select_input(data, spec['input']))
                item['rows'].extend(collect_predictions(
                    prediction,
                    target,
                    visible_height=visible_height,
                    case_id=batch.get('case_id'),
                    seq_name=batch.get('seq_name'),
                    pred_mode=cfg['model']['mode'],
                    dT=model.dT,
                ))

            if args.max_batches is not None and batch_idx + 1 >= args.max_batches:
                break

    output_dir = Path(args.output_dir)
    combined = []
    for key, item in models.items():
        spec = item['spec']
        out_dir = output_dir / key
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(item['rows'])
        df.to_csv(out_dir / 'eval_results_sorted.csv', index=False)
        summary = summarize_results(df)
        summary.insert(0, 'variant', key)
        summary.insert(1, 'experimental_setting', spec['label'])
        summary.to_csv(out_dir / 'summary.csv', index=False)
        combined.append(summary)
        print(f"\n[{key}] {spec['label']}")
        print_summary(summary)

    combined_df = pd.concat(combined, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_dir / 'summary.csv', index=False)
    print(f'\nSaved ablation outputs to {output_dir}')


if __name__ == '__main__':
    main()
