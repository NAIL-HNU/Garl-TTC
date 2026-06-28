#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train Garl-TTC / NNTTC.')
    parser.add_argument('--config', default=str(ROOT / 'configs/garl_ttc_eventdecoder.yaml'))
    parser.add_argument('--data-root', default=None, help='Dataset root containing data_blobs/.')
    parser.add_argument('--train-annotation-dir', default=None)
    parser.add_argument('--garlttc-annotation-root', default=None, help='HF-style GarlTTC dataset root containing data/, annotations/, and splits/.')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, cuda:0, ...')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--max-batches', type=int, default=None, help='Debug option for a short smoke run.')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from garl_ttc.config import load_config, set_config_value
    from garl_ttc.engine.trainer import train
    cfg = load_config(
        args.config,
        data_root=args.data_root,
        output_dir=args.output_dir,
        train_annotation_dir=args.train_annotation_dir,
        garlttc_annotation_root=args.garlttc_annotation_root,
    )
    if args.batch_size is not None:
        set_config_value(cfg, 'training_settings.batch_size', args.batch_size)
    if args.num_workers is not None:
        set_config_value(cfg, 'training_settings.num_threads', args.num_workers)
    train(
        cfg,
        output_dir=args.output_dir,
        device=args.device,
        epochs=args.epochs,
        max_batches=args.max_batches,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
