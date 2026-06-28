#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable


EXPECTED_FORMAT = 'garlttc_prediction_v1'
SCORE_KEYS = (
    'MiDc', 'FRc',
    'MiDs', 'FRs',
    'MiDl', 'FRl',
    'MiDn', 'FRn',
    'overall_MiD', 'mean_MiD', 'failed_rate', 'num_samples',
)
PAPER_MID_WEIGHTS = {'c': 0.5, 's': 0.3, 'l': 0.1, 'n': 0.1}
IGNORED_SUBMISSION_FILES = {'metadata'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CodaBench scorer for the GarlTTC benchmark.')
    parser.add_argument('input_dir', nargs='?', type=Path)
    parser.add_argument('output_dir', nargs='?', type=Path)
    parser.add_argument('--submission', type=Path, default=None)
    parser.add_argument('--submission-zip', type=Path, default=None)
    parser.add_argument('--reference', type=Path, default=None)
    parser.add_argument('--reference-zip', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--dt', type=float, default=0.1)
    return parser.parse_args()


def extract_zip(zip_path: Path, dst: Path) -> Path:
    if not zip_path.exists():
        raise FileNotFoundError(f'Missing zip file: {zip_path}')
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dst)
    except zipfile.BadZipFile as exc:
        raise ValueError(f'Invalid zip file: {zip_path}') from exc
    return dst


def expand_if_single_zip(path: Path, dst: Path) -> Path:
    if path.is_file() and path.suffix.lower() == '.zip':
        return extract_zip(path, dst)
    if path.is_dir():
        zip_files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == '.zip')
        direct_jsons = list(path.rglob('submission.json')) + list(path.rglob('labels.json'))
        if len(zip_files) == 1 and not direct_jsons:
            return extract_zip(zip_files[0], dst)
    return path


def resolve_codabench_input_dirs(input_dir: Path) -> tuple[Path, Path]:
    submission_candidates = [input_dir / 'res', input_dir / 'submission', input_dir]
    reference_candidates = [input_dir / 'ref', input_dir / 'reference', input_dir / 'reference_data', input_dir]
    submission_root = next((path for path in submission_candidates if path.exists()), input_dir)
    reference_root = next((path for path in reference_candidates if path.exists()), input_dir)
    return submission_root, reference_root


def require_single_submission_json(root: Path) -> Path:
    if root.is_file() and root.name == 'submission.json':
        return root
    if root.is_file():
        raise ValueError('GarlTTC benchmark only accepts JSON submissions named submission.json.')
    files = sorted(p for p in root.rglob('*') if p.is_file()) if root.exists() else []
    files = [
        p for p in files
        if p.relative_to(root).parts[0] not in IGNORED_SUBMISSION_FILES
    ]
    if any(p.suffix.lower() == '.csv' for p in files):
        raise ValueError('GarlTTC benchmark only accepts JSON submissions named submission.json.')
    expected = root / 'submission.json'
    if files != [expected]:
        rels = [str(p.relative_to(root)) for p in files[:10]]
        raise ValueError(
            'GarlTTC benchmark expects exactly one root-level submission.json and no extra files. '
            f'Found: {rels}'
        )
    return expected


def require_single_reference_json(root: Path) -> Path:
    if root.is_file() and root.name == 'labels.json':
        return root
    matches = sorted(root.rglob('labels.json')) if root.exists() else []
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one reference labels.json, found {len(matches)}: {matches}')
    return matches[0]


def resolve_paths(args: argparse.Namespace, tmp_dir: Path) -> tuple[Path, Path, Path]:
    output_dir = args.output or args.output_dir or Path('/app/output')
    if args.submission or args.submission_zip:
        submission_root = args.submission or args.submission_zip
    elif args.input_dir:
        submission_root, _ = resolve_codabench_input_dirs(args.input_dir)
    else:
        submission_root = Path('/app/input/res')

    if args.reference or args.reference_zip:
        reference_root = args.reference or args.reference_zip
    elif args.input_dir:
        _, reference_root = resolve_codabench_input_dirs(args.input_dir)
    else:
        reference_root = Path('/app/input/ref')

    submission_root = expand_if_single_zip(submission_root, tmp_dir / 'submission')
    reference_root = expand_if_single_zip(reference_root, tmp_dir / 'reference')
    return require_single_submission_json(submission_root), require_single_reference_json(reference_root), output_dir


def load_submission(path: Path) -> dict[str, float]:
    if path.suffix.lower() != '.json':
        raise ValueError('GarlTTC benchmark only accepts JSON submissions named submission.json.')
    payload = json.loads(path.read_text(encoding='utf-8'))
    meta = payload.get('meta') or {}
    if meta.get('format') != EXPECTED_FORMAT:
        raise ValueError(f"submission meta.format must be {EXPECTED_FORMAT!r}")
    results = payload.get('results')
    if not isinstance(results, dict):
        raise ValueError('submission results must be an object keyed by sample_token.')

    out: dict[str, float] = {}
    for token, item in results.items():
        value = item.get('ttc') if isinstance(item, dict) else item
        if value is None:
            raise ValueError(f'Missing ttc for sample_token={token}')
        try:
            out[str(token)] = float(value)
        except Exception as exc:
            raise ValueError(f'Invalid TTC value for sample_token={token}: {value!r}') from exc
    return out


def load_reference(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    rows = payload.get('labels', payload)
    if not isinstance(rows, list):
        raise ValueError('reference labels.json must contain a labels list.')
    seen = set()
    for row in rows:
        if 'sample_token' not in row or 'ttc' not in row:
            raise ValueError('reference labels require sample_token and ttc fields.')
        token = str(row['sample_token'])
        if token in seen:
            raise ValueError(f'reference labels contain duplicate sample_token: {token}')
        seen.add(token)
    return rows


def finite_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else float('nan')


def bin_suffix(gt_ttc: float) -> str | None:
    if -10 < gt_ttc <= 0:
        return 'n'
    if 0 < gt_ttc <= 3:
        return 'c'
    if 3 < gt_ttc <= 6:
        return 's'
    if 6 < gt_ttc <= 10:
        return 'l'
    return None


def evaluate_rows(reference: list[dict[str, Any]], prediction: dict[str, float], *, dT: float) -> dict[str, float]:
    ref_tokens = {str(row['sample_token']) for row in reference}
    pred_tokens = set(prediction)
    missing = sorted(ref_tokens - pred_tokens)
    extra = sorted(pred_tokens - ref_tokens)
    if missing or extra:
        details = []
        if missing:
            details.append(f'missing={missing[:10]} count={len(missing)}')
        if extra:
            details.append(f'extra={extra[:10]} count={len(extra)}')
        raise ValueError('submission sample_token set mismatch: ' + '; '.join(details))

    groups = {suffix: [] for suffix in ('c', 's', 'l', 'n')}
    all_rows = []
    for ref in reference:
        token = str(ref['sample_token'])
        pred_ttc = float(prediction[token])
        gt_ttc = float(ref['ttc'])
        pred_height_ratio = 1.0 - (dT / pred_ttc) if pred_ttc not in (0.0, -0.0) else float('nan')
        gt_height_ratio = 1.0 - (dT / gt_ttc)
        if pred_height_ratio <= 0 or gt_height_ratio <= 0 or not math.isfinite(pred_height_ratio):
            mid = float('nan')
        else:
            mid = abs(math.log(gt_height_ratio) - math.log(pred_height_ratio)) * 1e4
        failed = (not math.isfinite(pred_ttc)) or abs(pred_ttc) < 0.1
        row = {
            'MiD': mid,
            'failed': failed,
            'gt_ttc': gt_ttc,
        }
        all_rows.append(row)
        suffix = bin_suffix(gt_ttc)
        if suffix is not None:
            groups[suffix].append(row)

    out: dict[str, float] = {}
    for suffix, rows in groups.items():
        if not rows:
            out[f'MiD{suffix}'] = float('nan')
            out[f'FR{suffix}'] = float('nan')
            continue
        out[f'MiD{suffix}'] = finite_mean(row['MiD'] for row in rows)
        out[f'FR{suffix}'] = sum(1 for row in rows if row['failed']) / len(rows) * 100.0

    out['mean_MiD'] = finite_mean(row['MiD'] for row in all_rows)
    out['overall_MiD'] = sum(out[f'MiD{suffix}'] * weight for suffix, weight in PAPER_MID_WEIGHTS.items())
    out['num_samples'] = len(all_rows)
    out['failed_rate'] = sum(1 for row in all_rows if row['failed']) / len(all_rows) * 100.0 if all_rows else float('nan')
    return out


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_outputs(scores: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'scores.json').write_text(json.dumps(json_safe(scores), indent=2) + '\n', encoding='utf-8')
    lines = [f"{key}: {'' if scores.get(key) is None else scores.get(key)}" for key in SCORE_KEYS]
    (output_dir / 'scores.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        try:
            submission_json, reference_json, output_dir = resolve_paths(args, tmp_dir)
            prediction = load_submission(submission_json)
            reference = load_reference(reference_json)
            scores = evaluate_rows(reference, prediction, dT=args.dt)
            write_outputs(scores, output_dir)
        except Exception as exc:
            output_dir = args.output or args.output_dir or Path('/app/output')
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / 'error.txt').write_text(str(exc) + '\n', encoding='utf-8')
            write_outputs({key: None for key in SCORE_KEYS}, output_dir)
            raise


if __name__ == '__main__':
    main()
