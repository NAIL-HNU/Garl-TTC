from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PAPER_MID_WEIGHTS = {'c': 0.5, 's': 0.3, 'l': 0.1, 'n': 0.1}
EXPECTED_FORMAT = 'garlttc_prediction_v1'


def _as_float(value: Any, *, token: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f'Invalid TTC value for sample_token={token}: {value!r}') from exc
    return out


def load_submission(path: str | Path) -> dict[str, float]:
    path = Path(path)
    if path.suffix.lower() != '.json':
        raise ValueError('GarlTTC benchmark only accepts JSON submissions.')
    payload = json.loads(path.read_text())
    meta = payload.get('meta') or {}
    if meta.get('format') != EXPECTED_FORMAT:
        raise ValueError(f"submission meta.format must be {EXPECTED_FORMAT!r}")
    results = payload.get('results')
    if not isinstance(results, dict):
        raise ValueError('submission results must be an object keyed by sample_token.')

    out: dict[str, float] = {}
    for token, item in results.items():
        if isinstance(item, dict):
            if 'ttc' not in item:
                raise ValueError(f'Missing ttc for sample_token={token}')
            value = item['ttc']
        else:
            value = item
        out[str(token)] = _as_float(value, token=str(token))
    return out


def load_reference(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():
        path = path / 'test/labels.parquet'
    df = pd.read_parquet(path)
    required = {'sample_token', 'ttc'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'reference labels missing columns: {sorted(missing)}')
    if df['sample_token'].duplicated().any():
        dupes = df.loc[df['sample_token'].duplicated(), 'sample_token'].head(10).tolist()
        raise ValueError(f'reference labels contain duplicate sample_token values: {dupes}')
    return df


def _rows_from_prediction(reference: pd.DataFrame, prediction: dict[str, float], *, strict_tokens: bool, dT: float) -> pd.DataFrame:
    ref_tokens = set(map(str, reference['sample_token']))
    pred_tokens = set(prediction)
    missing = sorted(ref_tokens - pred_tokens)
    extra = sorted(pred_tokens - ref_tokens)
    if strict_tokens and (missing or extra):
        details = []
        if missing:
            details.append(f'missing={missing[:10]} count={len(missing)}')
        if extra:
            details.append(f'extra={extra[:10]} count={len(extra)}')
        raise ValueError('submission sample_token set mismatch: ' + '; '.join(details))

    rows = []
    for ref in reference.to_dict('records'):
        token = str(ref['sample_token'])
        pred_ttc = prediction.get(token, float('nan'))
        gt_ttc = float(ref['ttc'])
        pred_height_ratio = 1.0 - (dT / pred_ttc) if pred_ttc not in (0.0, -0.0) else float('nan')
        gt_height_ratio = 1.0 - (dT / gt_ttc)
        if pred_height_ratio <= 0 or gt_height_ratio <= 0 or not math.isfinite(pred_height_ratio):
            mid = float('nan')
        else:
            mid = abs(math.log(gt_height_ratio) - math.log(pred_height_ratio)) * 1e4
        rows.append({
            'sample_token': token,
            'sequence_id': ref.get('sequence_id'),
            'pred_ttc': pred_ttc,
            'gt_ttc': gt_ttc,
            'pred_height_ratio': pred_height_ratio,
            'gt_height_ratio_from_gtttc': gt_height_ratio,
            'MiD': mid,
        })
    return pd.DataFrame(rows)


def summarize_results(df: pd.DataFrame) -> dict[str, float]:
    df = df.copy()
    bins = [-10, 0, 3, 6, 10]
    labels = {
        pd.Interval(0, 3, closed='right'): 'c',
        pd.Interval(3, 6, closed='right'): 's',
        pd.Interval(6, 10, closed='right'): 'l',
        pd.Interval(-10, 0, closed='right'): 'n',
    }
    df['gt_ttc_bins'] = pd.cut(df['gt_ttc'], bins=bins)
    failed = df['pred_ttc'].isna() | np.isinf(df['pred_ttc']) | (np.abs(df['pred_ttc']) < 0.1)
    df['_failed'] = failed

    out: dict[str, float] = {}
    for interval, suffix in labels.items():
        group = df[df['gt_ttc_bins'] == interval]
        if len(group) == 0:
            out[f'MiD{suffix}'] = float('nan')
            out[f'FR{suffix}'] = float('nan')
            continue
        out[f'MiD{suffix}'] = float(group['MiD'].mean())
        out[f'FR{suffix}'] = float(group['_failed'].mean() * 100.0)

    out['mean_MiD'] = float(df['MiD'].mean())
    out['overall_MiD'] = float(sum(out[f'MiD{suffix}'] * weight for suffix, weight in PAPER_MID_WEIGHTS.items()))
    out['num_samples'] = int(len(df))
    out['failed_rate'] = float(df['_failed'].mean() * 100.0 if len(df) else np.nan)
    return out


def evaluate_submission(
    *,
    submission_path: str | Path,
    reference_path: str | Path,
    output_dir: str | Path | None = None,
    strict_tokens: bool = True,
    dT: float = 0.1,
) -> dict[str, float]:
    prediction = load_submission(submission_path)
    reference = load_reference(reference_path)
    rows = _rows_from_prediction(reference, prediction, strict_tokens=strict_tokens, dT=dT)
    summary = summarize_results(rows)

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rows.to_csv(out_dir / 'predictions_with_gt.csv', index=False)
        (out_dir / 'scores.json').write_text(json.dumps(summary, indent=2) + '\n')
        lines = ['GarlTTC benchmark scores']
        for key in ['MiDc', 'FRc', 'MiDs', 'FRs', 'MiDl', 'FRl', 'MiDn', 'FRn', 'overall_MiD', 'num_samples', 'failed_rate']:
            lines.append(f'{key}: {summary[key]}')
        (out_dir / 'scores.txt').write_text('\n'.join(lines) + '\n')
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate a GarlTTC JSON submission against private labels.')
    parser.add_argument('--submission', required=True)
    parser.add_argument('--reference', required=True, help='private_labels root or labels.parquet.')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--non-strict-tokens', action='store_true')
    parser.add_argument('--dt', type=float, default=0.1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = evaluate_submission(
        submission_path=args.submission,
        reference_path=args.reference,
        output_dir=args.output_dir,
        strict_tokens=not args.non_strict_tokens,
        dT=args.dt,
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
