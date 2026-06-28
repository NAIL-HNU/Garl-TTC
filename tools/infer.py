#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run Garl-TTC inference and write per-window TTC predictions.')
    parser.add_argument('--config', default=str(ROOT / 'configs/garl_ttc_eventdecoder.yaml'))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default=None, help='Dataset root containing data_blobs/.')
    parser.add_argument('--annotation-dir', default=None, help='Annotation directory for the inference assets.')
    parser.add_argument('--garlttc-annotation-root', default=None, help='HF-style GarlTTC dataset root containing public test inputs.')
    parser.add_argument('--asset-file', default=None, help='Asset list to run inference on. Defaults to benchmark test assets.')
    parser.add_argument('--output-dir', default=str(ROOT / 'outputs/infer'))
    parser.add_argument('--output-json', default=None, help='Write a CodaBench JSON submission to this path.')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--max-batches', type=int, default=None)
    parser.add_argument('--seq-ts', type=int, default=5)
    parser.add_argument('--non-strict', action='store_true', help='Load checkpoint with strict=False.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from garl_ttc.config import load_config, set_config_value
    from garl_ttc.engine.evaluator import evaluate
    cfg = load_config(
        args.config,
        data_root=args.data_root,
        test_annotation_dir=args.annotation_dir,
        garlttc_annotation_root=args.garlttc_annotation_root,
    )
    if args.asset_file is not None:
        set_config_value(cfg, 'dataset.test.asset_path', args.asset_file)
    if args.garlttc_annotation_root is not None or args.output_json is not None:
        from garl_ttc.engine.inference import run_inference
        output_json = args.output_json or str(Path(args.output_dir) / 'submission.json')
        run_inference(
            cfg,
            args.checkpoint,
            output_json,
            device=args.device,
            max_batches=args.max_batches,
            seq_ts=args.seq_ts,
            strict=not args.non_strict,
        )
        return
    evaluate(
        cfg,
        args.checkpoint,
        args.output_dir,
        device=args.device,
        max_batches=args.max_batches,
        seq_ts=args.seq_ts,
        strict=not args.non_strict,
    )


if __name__ == '__main__':
    main()
