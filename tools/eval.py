#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate Garl-TTC / NNTTC on the TTC benchmark split.')
    parser.add_argument('--config', default=str(ROOT / 'configs/garl_ttc_eventdecoder.yaml'))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default=None, help='Dataset root containing data_blobs/.')
    parser.add_argument('--test-annotation-dir', default=None)
    parser.add_argument('--garlttc-annotation-root', default=None, help='HF-style GarlTTC dataset root containing data/ and splits/.')
    parser.add_argument('--test-labels-parquet', default=None, help='Private labels parquet for HF-style evaluation.')
    parser.add_argument('--asset-file', default=None, help='Override benchmark asset list.')
    parser.add_argument('--output-dir', default=str(ROOT / 'outputs/eval'))
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, cuda:0, ...')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--max-batches', type=int, default=None, help='Debug option for a short smoke run.')
    parser.add_argument('--seq-ts', type=int, default=5)
    parser.add_argument('--ttc-min', type=float, default=-10)
    parser.add_argument('--ttc-max', type=float, default=10)
    parser.add_argument('--non-strict', action='store_true', help='Load checkpoint with strict=False.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from garl_ttc.config import load_config, set_config_value
    from garl_ttc.engine.evaluator import evaluate
    cfg = load_config(
        args.config,
        data_root=args.data_root,
        test_annotation_dir=args.test_annotation_dir,
        garlttc_annotation_root=args.garlttc_annotation_root,
        test_labels_parquet=args.test_labels_parquet,
    )
    if args.asset_file is not None:
        set_config_value(cfg, 'dataset.test.asset_path', args.asset_file)
    if args.batch_size is not None:
        set_config_value(cfg, 'testing_settings.batch_size', args.batch_size)
    if args.num_workers is not None:
        set_config_value(cfg, 'testing_settings.num_threads', args.num_workers)
    evaluate(
        cfg,
        args.checkpoint,
        args.output_dir,
        device=args.device,
        max_batches=args.max_batches,
        seq_ts=args.seq_ts,
        test_ttc_range=(args.ttc_min, args.ttc_max),
        strict=not args.non_strict,
    )


if __name__ == '__main__':
    main()
