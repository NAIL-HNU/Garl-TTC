from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_CHECKPOINTS = [
    'paper_ours_full.pth',
    'paper_event_only_lhr.pth',
    'paper_visual_only_lhr.pth',
    'paper_multi_modal_lhr_late_fusion.pth',
    'paper_multi_modal_lhr_early_fusion.pth',
    'paper_event_only_baseline.pth',
    'paper_visual_only_baseline.pth',
    'paper_multi_modal_baseline.pth',
]
PAPER_TEST12_METRICS = {
    'benchmark': 'GarlTTC test12',
    'num_samples': 6762,
    'method': 'Garl-TTC (Ours)',
    'type': 'L',
    'modality': 'E+V',
    'metrics': {
        'MiDc': 53.1,
        'FRc': 0.0,
        'RTEc': 16.6,
        'MiDs': 37.6,
        'FRs': 0.0,
        'RTEs': 20.0,
        'MiDl': 40.6,
        'FRl': 0.0,
        'RTEl': 34.1,
        'MiDn': 31.3,
        'FRn': 0.0,
        'RTEn': 28.2,
    },
    'metric_bins': {
        'c': '(0, 3]',
        's': '(3, 6]',
        'l': '(6, 10]',
        'n': '(-10, 0]',
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Stage the local GarlTTC Hugging Face model repository.')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--repo-id', default='NAIL-HNU/GarlTTC-model')
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints'))
    parser.add_argument('--config-dir', type=Path, default=Path('configs'))
    parser.add_argument('--metrics-dir', type=Path, default=Path('metrics'))
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def stage_model_repo(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_root
    if root.exists() and args.overwrite:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'checkpoints').mkdir(exist_ok=True)
    (root / 'configs').mkdir(exist_ok=True)
    (root / 'metrics').mkdir(exist_ok=True)

    checkpoints = []
    missing = []
    for name in DEFAULT_CHECKPOINTS:
        src = args.checkpoint_dir / name
        if not src.is_file():
            missing.append(str(src))
            continue
        dst = root / 'checkpoints' / name
        shutil.copy2(src, dst)
        checkpoints.append({
            'name': name,
            'path': f'checkpoints/{name}',
            'bytes': dst.stat().st_size,
            'sha256': sha256(dst),
        })
    if missing:
        raise FileNotFoundError('Missing checkpoint(s): ' + ', '.join(missing))

    for src in sorted(args.config_dir.glob('*.yaml')):
        shutil.copy2(src, root / 'configs' / src.name)
    ablation_src = args.config_dir / 'ablation'
    if ablation_src.is_dir():
        shutil.copytree(ablation_src, root / 'configs' / 'ablation', dirs_exist_ok=True)
    if args.metrics_dir.is_dir():
        for src in sorted(args.metrics_dir.glob('*.json')):
            shutil.copy2(src, root / 'metrics' / src.name)
    write_json(PAPER_TEST12_METRICS, root / 'metrics' / 'paper_test12.json')

    readme = f"""---
library_name: pytorch
datasets:
- NAIL-HNU/eAP-dataset
- NAIL-HNU/GarlTTC-dataset
tags:
- ttc
- time-to-collision
- event-camera
- rgb-event
- multimodal
- autonomous-driving
- pytorch
---

# GarlTTC Model

This repository stages checkpoints for the GarlTTC release.

Dataset dependencies:

- `NAIL-HNU/eAP-dataset`
- `NAIL-HNU/GarlTTC-dataset`

Primary checkpoint:

- `checkpoints/paper_ours_full.pth`

Reference metrics are stored in `metrics/paper_test12.json`. If a local release
verification run is available, its raw output can be staged as `metrics/test.json`.

Ablation checkpoints are included for paper table reproduction. Use the matching
YAML files under `configs/`.
"""
    (root / 'README.md').write_text(readme, encoding='utf-8')
    metadata = {
        'repo_id': args.repo_id,
        'checkpoints': checkpoints,
        'configs': sorted(str(path.relative_to(root)) for path in (root / 'configs').rglob('*') if path.is_file()),
        'metrics': sorted(str(path.relative_to(root)) for path in (root / 'metrics').glob('*.json')),
    }
    write_json(metadata, root / 'model_release_metadata.json')
    return metadata


def main(argv: list[str] | None = None) -> None:
    metadata = stage_model_repo(parse_args(argv))
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
