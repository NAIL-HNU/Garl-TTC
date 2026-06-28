#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate SAM foreground masks for Garl-TTC training.')
    parser.add_argument('--config', default=str(ROOT / 'configs/garl_ttc_eventdecoder.yaml'))
    parser.add_argument('--data-root', default=None, help='Dataset root containing data_blobs/.')
    parser.add_argument('--train-annotation-dir', default=None)
    parser.add_argument('--test-annotation-dir', default=None)
    parser.add_argument('--split', choices=['train', 'test', 'all'], default='train')
    parser.add_argument('--sam-checkpoint', default=str(ROOT / 'checkpoints/sam_vit_h_4b8939.pth'))
    parser.add_argument('--model-type', default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'])
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, cuda:0, ...')
    parser.add_argument('--scale-factor', type=float, default=1.0)
    parser.add_argument('--box-expand', type=float, default=1.1)
    parser.add_argument('--image-color', choices=['bgr', 'rgb'], default='bgr')
    parser.add_argument('--combine-mode', choices=['last', 'union'], default='last')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--vis-dir', default=str(ROOT / 'outputs/sam_mask_vis'))
    parser.add_argument('--limit', type=int, default=None, help='Debug option for a short run.')
    parser.add_argument('--seq-ts', type=int, default=5)
    parser.add_argument('--ttc-min', type=float, default=-10)
    parser.add_argument('--ttc-max', type=float, default=10)
    return parser.parse_args()


def build_dataset(cfg: dict, split: str, args: argparse.Namespace):
    from garl_ttc.datasets import TTCEstimationDataset

    if split == 'train':
        return TTCEstimationDataset(cfg, 'train', seed=0, db_mode='window')
    if split == 'test':
        return TTCEstimationDataset(
            cfg,
            'test',
            seed=0,
            db_mode='sequence',
            seq_ts=args.seq_ts,
            test_ttc_range=[args.ttc_min, args.ttc_max],
        )
    raise ValueError(f'Unsupported split: {split}')


def main() -> None:
    args = parse_args()

    from garl_ttc.config import load_config
    from garl_ttc.engine.runtime import pick_device
    from garl_ttc.utils.sam_masks import generate_sam_masks_for_dataset

    cfg = load_config(
        args.config,
        data_root=args.data_root,
        train_annotation_dir=args.train_annotation_dir,
        test_annotation_dir=args.test_annotation_dir,
    )
    device = pick_device(args.device)
    splits = ['train', 'test'] if args.split == 'all' else [args.split]

    for split in splits:
        dataset = build_dataset(cfg, split, args)
        stats = generate_sam_masks_for_dataset(
            dataset,
            args.sam_checkpoint,
            model_type=args.model_type,
            device=str(device),
            scale_factor=args.scale_factor,
            expand=args.box_expand,
            image_color=args.image_color,
            combine_mode=args.combine_mode,
            overwrite=args.overwrite,
            visualize=args.visualize,
            vis_dir=args.vis_dir,
            limit=args.limit,
        )
        print(f'[{split}] {stats}')


if __name__ == '__main__':
    main()
