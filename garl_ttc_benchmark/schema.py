from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


TEST_GT_FORBIDDEN_COLUMNS = {
    'ttc',
    'gt_ttc',
    'ttc_2d',
    'ttc_3d',
    'frame_ttc',
    'height_ratio',
    'gt_height_ratio',
    'visible_height',
    'gt_visible_height',
    'dimension',
    'gt_dimension',
    'depth',
    'gt_depth',
    'depth_start_m',
    'depth_end_m',
    'box3d_h',
    'box3d_Fcam',
    'box3d_fcam',
}


def read_lines(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def write_lines(path: str | Path, values: Iterable[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(f'{value}\n' for value in values))


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def normalize_dataset_split(value: str | None) -> str | None:
    if value is None:
        return None
    if value in {'training', 'train'}:
        return 'train'
    if value in {'benchmark_test', 'test'}:
        return 'test'
    return value


def sample_token(public_track_id: str, timestamp_us: int, seq_name: str | None = None) -> str:
    case_id = f'{public_track_id}_{int(timestamp_us)}'
    if seq_name:
        return f'{seq_name}_{case_id}'
    return case_id


def assert_unique(values: Iterable[str], *, name: str) -> None:
    seen: set[str] = set()
    dupes: set[str] = set()
    for value in values:
        if value in seen:
            dupes.add(value)
        seen.add(value)
    if dupes:
        preview = ', '.join(sorted(dupes)[:10])
        raise ValueError(f'{name} contains duplicate values: {preview}')
