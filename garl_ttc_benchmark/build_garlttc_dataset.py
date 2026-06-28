from __future__ import annotations

import argparse
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from garl_ttc.utils.io import read_pkl
from garl_ttc_benchmark.schema import (
    TEST_GT_FORBIDDEN_COLUMNS,
    assert_unique,
    normalize_dataset_split,
    read_json,
    read_lines,
    sample_token,
    write_json,
    write_lines,
)


DEFAULT_TRAIN_SPLIT = Path('configs/splits/train.txt')
DEFAULT_TEST_SPLIT = Path('configs/splits/test.txt')
DEFAULT_WINDOW_INTERVAL = 1
DEFAULT_SEQ_TS = 5
DEFAULT_TEST_TTC_RANGE = (-10.0, 10.0)
DEFAULT_EXPECTED_TRAIN_ASSETS = 40
DEFAULT_EXPECTED_TEST_ASSETS = 12
DEFAULT_EXPECTED_TEST_ROWS = 6762


@dataclass(frozen=True)
class MediaRef:
    events_path: str
    rgb_shard_path: str
    rgb_member_path: str


def _as_list(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_as_list(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _timestamp_key(timestamp_us: int) -> str:
    ts_s_part = int(timestamp_us) // int(1e6)
    ts_us_part = int((int(timestamp_us) % int(1e6)) * 1e3)
    return f'{ts_s_part:012d}_{ts_us_part:012d}'


def _sliding_window(values: list[int], window_size: int, step_size: int = 1):
    for i in range(0, len(values) - window_size + 1, step_size):
        yield values[i:i + window_size]


def _continuous_sequences(valid_ts_list: list[int]) -> list[list[int]]:
    valid_ts_list = sorted(int(v) for v in valid_ts_list)
    if not valid_ts_list:
        return []
    start_i = 0
    out: list[list[int]] = []
    for index in range(1, len(valid_ts_list)):
        if valid_ts_list[index] - valid_ts_list[index - 1] > 100000:
            out.append(valid_ts_list[start_i:index])
            start_i = index
    # Match TTCEstimationDataset.get_test_continue_sequence_from_one_track exactly:
    # when a discontinuity exists, the legacy loader does not append the final
    # trailing segment. Paper test12 metrics and sample count depend on this.
    if len(out) == 0:
        out.append(valid_ts_list)
    return [seq for seq in out if seq]


def _selected_indices(sync: str, window_interval: int, n: int) -> list[int]:
    indices = list(range(n))
    if sync == 'front':
        return indices[1:-window_interval] + indices[-1:]
    if sync == 'back':
        return indices[:-window_interval - 1] + indices[-2:-1]
    raise ValueError(f'Unsupported sync: {sync}')


def _load_media_index(eap_roots: list[Path]) -> dict[tuple[str, int], MediaRef]:
    index: dict[tuple[str, int], MediaRef] = {}
    for root in eap_roots:
        if not root:
            continue
        root = Path(root)
        for parquet in (root / 'data').glob('*.parquet'):
            if not parquet.is_file():
                continue
            df = pd.read_parquet(parquet)
            for row in df.to_dict('records'):
                seq = row.get('sequence_id')
                ts = row.get('rgb_exposure_start_timestamp_us')
                if seq is None or ts is None:
                    continue
                index[(str(seq), int(ts))] = MediaRef(
                    events_path=str(row.get('events_path') or ''),
                    rgb_shard_path=str(row.get('rgb_shard_path') or ''),
                    rgb_member_path=str(row.get('rgb_member_path') or ''),
                )
    return index


def _fallback_media_ref(split: str, sequence_id: str, image_path: str) -> MediaRef:
    filename = Path(image_path).name
    return MediaRef(
        events_path=f'data/{split}/{sequence_id}/events.h5',
        rgb_shard_path='',
        rgb_member_path=f'rgb/{filename}',
    )


def _frame_media_ref(
    *,
    media_index: dict[tuple[str, int], MediaRef],
    split: str,
    sequence_id: str,
    anno: dict,
    allow_missing_media_index: bool,
) -> MediaRef:
    ts = int(anno['meta']['corr_exposure_start_timestamp_us'])
    ref = media_index.get((sequence_id, ts))
    if ref is not None:
        return ref
    if allow_missing_media_index:
        return _fallback_media_ref(split, sequence_id, anno['meta']['image_path'])
    raise KeyError(
        f'Media index missing sequence={sequence_id} exposure_start={ts}. '
        'Build final eAP train/test first or pass --allow-missing-media-index for schema-only checks.'
    )


def _box3d_h(anno: dict) -> float:
    box3d_ego = anno.get('box3d_ego')
    if box3d_ego is None:
        return float('nan')
    return float(_as_list(box3d_ego)[5])


def _record_row(
    *,
    split: str,
    sequence_id: str,
    record: list[dict],
    media_index: dict[tuple[str, int], MediaRef],
    sync: str,
    window_interval: int,
    allow_missing_media_index: bool,
    include_labels: bool,
) -> tuple[dict, dict | None]:
    selected = [record[i] for i in _selected_indices(sync, window_interval, len(record))]
    target = selected[-1]
    token = sample_token(str(target['public_track_id']), int(target['timestamp']), target.get('seq_name'))

    refs = [
        _frame_media_ref(
            media_index=media_index,
            split=split,
            sequence_id=sequence_id,
            anno=anno,
            allow_missing_media_index=allow_missing_media_index,
        )
        for anno in selected
    ]
    source_starts = [int(anno['meta']['corr_exposure_start_timestamp_us']) for anno in record]
    event_windows = [
        [source_starts[0], source_starts[-2]],
        [source_starts[1], source_starts[-1]],
    ]
    data_row = {
        'sequence_id': sequence_id,
        'sample_token': token,
        'track_id': str(target.get('track_id') or target.get('public_track_id')),
        'public_track_id': str(target['public_track_id']),
        'timestamp_us': int(target['timestamp']),
        'seq_name': str(target.get('seq_name') or ''),
        'frame_timestamps_us': [int(anno['timestamp']) for anno in selected],
        'rgb_shard_paths': [ref.rgb_shard_path for ref in refs],
        'rgb_member_paths': [ref.rgb_member_path for ref in refs],
        'events_path': refs[-1].events_path,
        'event_windows_us': event_windows,
        'image_paths': [str(anno['meta']['image_path']) for anno in selected],
        'boxes_xyxy': [[int(v) for v in anno['box']] for anno in selected],
        'mask_paths': [str(anno['meta']['image_path']).replace('.png', '.npy') for anno in selected],
    }
    if not include_labels:
        return data_row, None

    label_row = {
        'sequence_id': sequence_id,
        'sample_token': token,
        'track_id': str(target.get('track_id') or target.get('public_track_id')),
        'public_track_id': str(target['public_track_id']),
        'timestamp_us': int(target['timestamp']),
        'ttc': float(target['ttc']),
        'frame_ttc': [float(anno['ttc']) for anno in selected],
        'box3d_h': float(_box3d_h(target)),
        'box3d_Fcam': [_as_list(anno.get('box3d_Fcam')) for anno in selected],
    }
    return data_row, label_row


def _train_records(track: dict, *, gt_range: tuple[float, float], window_interval: int) -> list[list[dict]]:
    valid_ts_list = sorted(int(v) for v in track.get('valid_ts_list') or [])
    if not valid_ts_list:
        return []
    track_anno = track.get('anno') or {}
    first = next(iter(track_anno.values()), None)
    if first is None:
        return []
    box3d_h = _box3d_h(first)
    if not np.isfinite(box3d_h) or box3d_h < 0 or box3d_h > 10:
        return []

    out: list[list[dict]] = []
    window_size = window_interval + 2
    for window in _sliding_window(valid_ts_list, window_size):
        if max(abs(window[i] - window[i + 1]) for i in range(len(window) - 1)) > 100000:
            continue
        record: list[dict] = []
        skip = False
        for ts in window:
            anno = dict(track_anno[_timestamp_key(ts)])
            ttc = float(anno.get('ttc'))
            if ttc < gt_range[0] or ttc > gt_range[1]:
                skip = True
                break
            anno['track_id'] = track.get('track_id')
            anno['box3d_h'] = box3d_h
            record.append(anno)
        if not skip:
            out.append(record)
    return out


def _test_sequences(
    track: dict,
    *,
    seq_ts: int,
    ttc_range: tuple[float, float],
) -> list[list[dict]]:
    track_anno = track.get('anno') or {}
    first = next(iter(track_anno.values()), None)
    if first is None:
        return []
    box3d_h = _box3d_h(first)
    if not np.isfinite(box3d_h) or box3d_h < 0 or box3d_h > 10:
        return []

    out: list[list[dict]] = []
    for cnt_seq in _continuous_sequences(track.get('valid_ts_list') or []):
        if len(cnt_seq) < seq_ts * 10 + 2:
            continue
        window_size = seq_ts * 10 + 1
        for window in _sliding_window(cnt_seq, window_size=window_size, step_size=2 * 10):
            if len(window) < window_size:
                continue
            annos: list[dict] = []
            out_range_count = 0
            t_min_us_list: list[int] = []
            for ts in window:
                anno = track_anno[_timestamp_key(ts)]
                t_min_us_list.append(int(anno['meta']['corr_exposure_start_timestamp_us']))
            seq_name = f"{track.get('track_id')}_{t_min_us_list[0]}_{t_min_us_list[-1]}"

            for ts in window:
                anno = dict(track_anno[_timestamp_key(ts)])
                anno['track_id'] = track.get('track_id')
                anno['box3d_h'] = box3d_h
                anno['seq_name'] = seq_name
                ttc = float(anno.get('ttc'))
                if ttc < ttc_range[0] or ttc > ttc_range[1] or ttc == 0:
                    out_range_count += 1
                annos.append(anno)
            if out_range_count <= 0.05 * window_size:
                out.append(annos)
    return out


def _test_records(
    track: dict,
    *,
    seq_ts: int,
    ttc_range: tuple[float, float],
    window_interval: int,
) -> list[list[dict]]:
    records: list[list[dict]] = []
    window_size = window_interval + 2
    for sequence in _test_sequences(track, seq_ts=seq_ts, ttc_range=ttc_range):
        for record in _sliding_window(sequence, window_size):
            if max(abs(record[i]['timestamp'] - record[i + 1]['timestamp']) for i in range(len(record) - 1)) > 100000:
                continue
            records.append(record)
    return records


def _rows_from_assets(
    *,
    annotation_dir: Path,
    assets: list[str],
    split: str,
    media_index: dict[tuple[str, int], MediaRef],
    sync: str,
    window_interval: int,
    allow_missing_media_index: bool,
    include_labels: bool,
    seq_ts: int,
    ttc_range: tuple[float, float],
    skip_missing_media: bool = False,
) -> tuple[list[dict], list[dict], int]:
    data_rows: list[dict] = []
    label_rows: list[dict] = []
    skipped_missing_media = 0
    for asset in assets:
        pkl_path = annotation_dir / f'{asset}.pkl'
        if not pkl_path.is_file():
            raise FileNotFoundError(pkl_path)
        for track in read_pkl(str(pkl_path)):
            if split == 'train':
                records = _train_records(track, gt_range=ttc_range, window_interval=window_interval)
            else:
                records = _test_records(track, seq_ts=seq_ts, ttc_range=ttc_range, window_interval=window_interval)
            for record in records:
                try:
                    data_row, label_row = _record_row(
                        split=split,
                        sequence_id=asset,
                        record=record,
                        media_index=media_index,
                        sync=sync,
                        window_interval=window_interval,
                        allow_missing_media_index=allow_missing_media_index,
                        include_labels=include_labels,
                    )
                except KeyError as exc:
                    if skip_missing_media and 'Media index missing' in str(exc):
                        skipped_missing_media += 1
                        continue
                    raise
                data_rows.append(data_row)
                if label_row is not None:
                    label_rows.append(label_row)
    return data_rows, label_rows, skipped_missing_media


def _published_train_assets(
    *,
    all_train_assets: list[str],
    eap_roots: list[Path],
    train_assets_file: Path | None,
) -> list[str]:
    if train_assets_file is not None:
        explicit = read_lines(train_assets_file)
        missing = sorted(set(explicit) - set(all_train_assets))
        if missing:
            raise ValueError(f'Explicit train asset list contains assets outside train split: {missing[:10]}')
        return explicit
    eap_assets: set[str] = set()
    for root in eap_roots:
        data_train = Path(root) / 'data' / 'train'
        if data_train.is_dir():
            eap_assets.update(p.name for p in data_train.iterdir() if p.is_dir())
    if eap_assets:
        return [asset for asset in all_train_assets if asset in eap_assets]
    raise ValueError(
        'No eAP public train assets found under data/train. '
        'Build/download eAP train40 first or pass --train-assets-file explicitly.'
    )


def _write_parquet(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_sample_submission(test_rows: list[dict], path: Path) -> None:
    sample = {
        'meta': {
            'format': 'garlttc_prediction_v1',
        },
        'results': {
            str(row['sample_token']): {
                'ttc': 1.0,
            }
            for row in test_rows
        },
    }
    write_json(path, sample)


def _write_dataset_card(public_root: Path) -> None:
    (public_root / 'README.md').write_text(
        """# GarlTTC Dataset

This repository contains the public GarlTTC structured annotations and benchmark
inputs. RGB frames and event streams are not duplicated here; they are referenced
from `NAIL-HNU/eAP-dataset`.

## Required Companion Dataset

Download `NAIL-HNU/eAP-dataset` first, then download this repository. Pass the
local eAP public root as `--data-root` and this repository root as
`--garlttc-annotation-root` when running the release code.

## Files

- `data/train.parquet`: train sample index and eAP media references. Samples
  whose frames are outside the published eAP train40 media are filtered out.
- `annotations/train.parquet`: train TTC supervision keyed by `sample_token`.
- `data/test_inputs.parquet`: public benchmark inputs with no TTC ground truth.
- `splits/train.txt`: the published train40 sequence list.
- `splits/test.txt`: the benchmark test12 sequence list.
- `dataset_info.json`: published sequence metadata.
- `sample_submission.json`: valid JSON submission template.

The benchmark accepts JSON only. Test labels are private and are not included in
this public dataset repository.
""",
        encoding='utf-8',
    )


def _validate_public(public_root: Path) -> dict:
    train = pd.read_parquet(public_root / 'data/train.parquet')
    train_labels = pd.read_parquet(public_root / 'annotations/train.parquet')
    test = pd.read_parquet(public_root / 'data/test_inputs.parquet')
    assert_unique(train['sample_token'], name='public train sample_token')
    assert_unique(train_labels['sample_token'], name='train annotations sample_token')
    assert_unique(test['sample_token'], name='public test sample_token')
    if set(train['sample_token']) != set(train_labels['sample_token']):
        raise ValueError('data/train.parquet and annotations/train.parquet sample_token sets differ')
    leaked = sorted(TEST_GT_FORBIDDEN_COLUMNS & set(test.columns))
    if leaked:
        raise ValueError(f'public test_inputs.parquet leaks GT-like columns: {leaked}')
    return {
        'train_rows': int(len(train)),
        'train_label_rows': int(len(train_labels)),
        'test_rows': int(len(test)),
        'train_sequences': sorted(map(str, train['sequence_id'].unique())),
        'test_sequences': sorted(map(str, test['sequence_id'].unique())),
    }


def _dataset_info_assets(dataset_info: dict, split: str) -> set[str]:
    return {
        str(asset)
        for asset, meta in dataset_info.items()
        if normalize_dataset_split((meta or {}).get('split')) == split
    }


def _assert_equal_set(name: str, actual: list[str] | set[str], expected: list[str] | set[str]) -> None:
    actual_set = set(actual)
    expected_set = set(expected)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise ValueError(f'{name} mismatch: missing={missing[:10]} extra={extra[:10]}')


def _assert_expected_count(name: str, actual: int, expected: int) -> None:
    if expected > 0 and actual != expected:
        raise ValueError(f'{name} expected {expected}, got {actual}')


def build_dataset(args: argparse.Namespace) -> dict:
    output_root = Path(args.output_root)
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f'{output_root} exists; pass --overwrite to replace it.')
        shutil.rmtree(output_root)
    public_root = output_root / 'public'
    private_root = output_root / 'private_labels'
    public_root.mkdir(parents=True)
    private_root.mkdir(parents=True)

    dataset_info = read_json(args.dataset_info)
    train_split_assets = read_lines(args.train_split)
    test_assets = read_lines(args.test_split)
    known_assets = set(dataset_info)
    missing_from_info = sorted((set(train_split_assets) | set(test_assets)) - known_assets)
    if missing_from_info:
        raise ValueError(f'Split files contain assets missing from dataset_info.json: {missing_from_info[:10]}')
    dataset_train_assets = _dataset_info_assets(dataset_info, 'train')
    dataset_test_assets = _dataset_info_assets(dataset_info, 'test')
    if dataset_train_assets:
        _assert_equal_set('source train split vs dataset_info train assets', train_split_assets, dataset_train_assets)
    if dataset_test_assets:
        _assert_equal_set('test split vs dataset_info test assets', test_assets, dataset_test_assets)

    eap_roots = [Path(p) for p in args.eap_public_root]
    train_assets = _published_train_assets(
        all_train_assets=train_split_assets,
        eap_roots=eap_roots,
        train_assets_file=Path(args.train_assets_file) if args.train_assets_file else None,
    )
    _assert_expected_count('published train assets', len(train_assets), args.expected_train_assets)
    _assert_expected_count('benchmark test assets', len(test_assets), args.expected_test_assets)
    media_index = _load_media_index(eap_roots)

    anno_root = Path(args.garlttc_annotation_root)
    train_data, train_labels, skipped_train_missing_media = _rows_from_assets(
        annotation_dir=anno_root / 'train',
        assets=train_assets,
        split='train',
        media_index=media_index,
        sync=args.sync,
        window_interval=args.window_interval,
        allow_missing_media_index=args.allow_missing_media_index,
        include_labels=True,
        seq_ts=args.seq_ts,
        ttc_range=tuple(args.train_ttc_range),
        skip_missing_media=not args.fail_on_missing_train_media,
    )
    test_data, test_labels, skipped_test_missing_media = _rows_from_assets(
        annotation_dir=anno_root / 'benchmark_test',
        assets=test_assets,
        split='test',
        media_index=media_index,
        sync=args.sync,
        window_interval=args.window_interval,
        allow_missing_media_index=args.allow_missing_media_index,
        include_labels=True,
        seq_ts=args.seq_ts,
        ttc_range=tuple(args.test_ttc_range),
        skip_missing_media=False,
    )
    if skipped_test_missing_media:
        raise RuntimeError('Internal error: test split must not skip missing media rows.')

    _write_parquet(train_data, public_root / 'data/train.parquet')
    _write_parquet(train_labels, public_root / 'annotations/train.parquet')
    _write_parquet(test_data, public_root / 'data/test_inputs.parquet')
    _write_parquet(test_labels, private_root / 'test/labels.parquet')
    write_lines(public_root / 'splits/train.txt', train_assets)
    write_lines(public_root / 'splits/test.txt', test_assets)
    _write_sample_submission(test_data, public_root / 'sample_submission.json')

    published_info = {
        asset: {**dataset_info.get(asset, {}), 'release_split': 'train'}
        for asset in train_assets
    }
    published_info.update({
        asset: {**dataset_info.get(asset, {}), 'release_split': 'test'}
        for asset in test_assets
    })
    write_json(public_root / 'dataset_info.json', published_info)
    _write_dataset_card(public_root)

    validation = _validate_public(public_root)
    private_labels = pd.read_parquet(private_root / 'test/labels.parquet')
    public_test = pd.read_parquet(public_root / 'data/test_inputs.parquet')
    private_tokens = private_labels['sample_token']
    if set(private_tokens) != set(public_test['sample_token']):
        raise ValueError('private test labels and public test inputs sample_token sets differ')
    _assert_equal_set('public train parquet sequences vs published train split', validation['train_sequences'], train_assets)
    _assert_equal_set('public test parquet sequences vs benchmark test split', validation['test_sequences'], test_assets)
    _assert_equal_set('private labels sequences vs benchmark test split', private_labels['sequence_id'].unique(), test_assets)
    _assert_expected_count('public test rows', validation['test_rows'], args.expected_test_rows)
    _assert_expected_count('private test rows', int(len(private_labels)), args.expected_test_rows)

    metadata = {
        'dataset_info_path': str(Path(args.dataset_info).resolve()),
        'source_annotation_root': str(anno_root.resolve()),
        'eap_public_roots': [str(p.resolve()) for p in eap_roots],
        'source_train_assets': len(train_split_assets),
        'published_train_assets': len(train_assets),
        'test_assets': test_assets,
        'splits': {
            'train': train_assets,
            'test': test_assets,
        },
        'validation': validation,
        'private_test_rows': int(len(test_labels)),
        'skipped_train_rows_missing_eap_media': int(skipped_train_missing_media),
        'expected': {
            'train_assets': args.expected_train_assets,
            'test_assets': args.expected_test_assets,
            'test_rows': args.expected_test_rows,
        },
        'window_interval': args.window_interval,
        'seq_ts': args.seq_ts,
        'sync': args.sync,
        'allow_missing_media_index': bool(args.allow_missing_media_index),
    }
    write_json(output_root / 'build_metadata.json', metadata)
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build the public/private GarlTTC Hugging Face dataset staging tree.')
    parser.add_argument('--dataset-info', required=True)
    parser.add_argument('--garlttc-annotation-root', required=True)
    parser.add_argument('--eap-public-root', action='append', required=True, help='One or more eAP HF public roots.')
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--train-split', default=str(DEFAULT_TRAIN_SPLIT))
    parser.add_argument('--test-split', default=str(DEFAULT_TEST_SPLIT))
    parser.add_argument('--train-assets-file', default=None, help='Optional explicit public train asset list.')
    parser.add_argument('--sync', default='front', choices=['front', 'back'])
    parser.add_argument('--window-interval', type=int, default=DEFAULT_WINDOW_INTERVAL)
    parser.add_argument('--seq-ts', type=int, default=DEFAULT_SEQ_TS)
    parser.add_argument('--train-ttc-range', type=float, nargs=2, default=[-10.0, 10.0])
    parser.add_argument('--test-ttc-range', type=float, nargs=2, default=list(DEFAULT_TEST_TTC_RANGE))
    parser.add_argument('--expected-train-assets', type=int, default=DEFAULT_EXPECTED_TRAIN_ASSETS)
    parser.add_argument('--expected-test-assets', type=int, default=DEFAULT_EXPECTED_TEST_ASSETS)
    parser.add_argument('--expected-test-rows', type=int, default=DEFAULT_EXPECTED_TEST_ROWS)
    parser.add_argument(
        '--fail-on-missing-train-media',
        action='store_true',
        help='Fail instead of filtering train samples that reference frames absent from the published eAP train media.',
    )
    parser.add_argument('--allow-missing-media-index', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    metadata = build_dataset(parse_args(argv))
    print(
        'Built GarlTTC dataset: '
        f"train_rows={metadata['validation']['train_rows']} "
        f"test_rows={metadata['validation']['test_rows']} "
        f"published_train_assets={metadata['published_train_assets']} "
        f"private_test_rows={metadata['private_test_rows']}"
    )


if __name__ == '__main__':
    main()
