from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open('r') as f:
        return yaml.safe_load(f)


def write_yaml(data: dict[str, Any], path: str | Path) -> None:
    with Path(path).open('w') as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _resolve_path(value: Any, base: Path) -> Any:
    if value in (None, 'None'):
        return value
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)
    return str((base / path).resolve())


def load_config(
    path: str | Path,
    *,
    data_root: str | None = None,
    output_dir: str | None = None,
    train_annotation_dir: str | None = None,
    test_annotation_dir: str | None = None,
    garlttc_annotation_root: str | None = None,
    test_labels_parquet: str | None = None,
) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    cfg = deepcopy(read_yaml(path))
    base = repo_root()
    cfg['config_path'] = str(path)

    if data_root is not None:
        cfg['dataset']['root'] = data_root
        cfg['dataset']['data_blob_dir'] = str(Path(data_root).expanduser() / 'data_blobs')
    if output_dir is not None:
        cfg.setdefault('dirs', {})['output'] = output_dir
    if train_annotation_dir is not None:
        cfg['dataset'].setdefault('train', {})['annotation_dir'] = train_annotation_dir
    if test_annotation_dir is not None:
        cfg['dataset'].setdefault('test', {})['annotation_dir'] = test_annotation_dir
    if garlttc_annotation_root is not None:
        ann_root = Path(garlttc_annotation_root).expanduser()
        cfg['dataset']['annotation_format'] = 'parquet'
        cfg['dataset'].setdefault('train', {}).update({
            'asset_path': str(ann_root / 'splits/train.txt'),
            'data_parquet': str(ann_root / 'data/train.parquet'),
            'labels_parquet': str(ann_root / 'annotations/train.parquet'),
        })
        cfg['dataset'].setdefault('test', {}).update({
            'asset_path': str(ann_root / 'splits/test.txt'),
            'data_parquet': str(ann_root / 'data/test_inputs.parquet'),
        })
    if test_labels_parquet is not None:
        cfg['dataset'].setdefault('test', {})['labels_parquet'] = test_labels_parquet

    if 'dirs' in cfg and 'output' in cfg['dirs']:
        cfg['dirs']['output'] = _resolve_path(cfg['dirs']['output'], base)

    dataset = cfg.get('dataset', {})
    for key in ('root', 'data_blob_dir', 'annotation_dir'):
        if key in dataset:
            dataset[key] = _resolve_path(dataset[key], base)
    for split in ('train', 'test', 'val'):
        split_cfg = dataset.get(split)
        if not split_cfg:
            continue
        for key in ('asset_path', 'annotation_dir', 'data_parquet', 'labels_parquet'):
            if key in split_cfg:
                split_cfg[key] = _resolve_path(split_cfg[key], base)

    model = cfg.get('model', {})
    for key in ('pretrained_ckpt_rgb', 'pretrained_ckpt_event'):
        if key in model:
            model[key] = _resolve_path(model[key], base)

    training = cfg.get('training_settings', {})
    if training.get('ckpt_path') not in (None, 'None'):
        training['ckpt_path'] = _resolve_path(training['ckpt_path'], base)

    return cfg


def set_config_value(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    node = cfg
    parts = dotted_key.split('.')
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value
