from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import torch


PAPER_MID_WEIGHTS = {'c': 0.5, 's': 0.3, 'l': 0.1, 'n': 0.1}


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _case_fields(case_pair, seq_pair=None) -> dict:
    case_1 = case_pair[0] if len(case_pair) > 0 else ''
    case_2 = case_pair[1] if len(case_pair) > 1 else case_1
    parts = case_2.split('_')
    trackid = '_'.join(parts[:2]) if len(parts) >= 2 else case_2
    try:
        tus = int(parts[-1])
    except ValueError:
        tus = -1
    if seq_pair is None:
        seq_name = ''
    elif isinstance(seq_pair, (list, tuple)) and seq_pair:
        seq_name = seq_pair[0]
    else:
        seq_name = seq_pair
    return {
        'seq_name': seq_name,
        'trackid': trackid,
        'tus': tus,
        'case_id_1': case_1,
        'case_id_2': case_2,
    }


def collect_predictions(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    visible_height: torch.Tensor | None,
    case_id: Iterable | None,
    seq_name: Iterable | None,
    pred_mode: str,
    dT: float,
) -> list[dict]:
    pred = to_numpy(prediction)
    gt_ttc = to_numpy(target).reshape(-1)
    n = len(gt_ttc)
    case_id = list(case_id) if case_id is not None else [['', ''] for _ in range(n)]
    seq_name = list(seq_name) if seq_name is not None else [None for _ in range(n)]

    if pred_mode == 'height_ratio':
        visual_heights = pred.reshape(n, -1)
        pred_height_ratio = visual_heights[:, 0] / visual_heights[:, 1]
        pred_ttc = dT / (1.0 - pred_height_ratio)
        gt_visible = to_numpy(visible_height).reshape(n, -1) if visible_height is not None else np.full((n, 2), np.nan)
    elif pred_mode == 'height_ratio_direct':
        pred_height_ratio = pred.reshape(n, -1)[:, 0]
        pred_ttc = dT / (1.0 - pred_height_ratio)
        visual_heights = np.full((n, 2), np.nan)
        gt_visible = np.full((n, 2), np.nan)
    elif pred_mode == 'baseline':
        pred_ttc = pred.reshape(n, -1)[:, 0]
        pred_height_ratio = 1.0 - (dT / pred_ttc)
        visual_heights = np.full((n, 2), np.nan)
        gt_visible = np.full((n, 2), np.nan)
    else:
        raise NotImplementedError(f'Unsupported pred_mode: {pred_mode}')

    gt_height_ratio = 1.0 - (dT / gt_ttc)
    ratio_error = np.abs(gt_height_ratio - pred_height_ratio)
    mid = np.abs(np.log(gt_height_ratio) - np.log(pred_height_ratio)) * 1e4
    rte = np.abs(pred_ttc - gt_ttc) / np.abs(gt_ttc) * 100.0

    rows = []
    for i in range(n):
        row = _case_fields(case_id[i], seq_name[i])
        row.update({
            'pred_height_ratio': pred_height_ratio[i],
            'gt_height_ratio_from_gtttc': gt_height_ratio[i],
            'pred_ttc': pred_ttc[i],
            'gt_ttc': gt_ttc[i],
            'ratio_error': ratio_error[i],
            'MiD': mid[i],
            'RTE': rte[i],
        })
        if pred_mode == 'height_ratio':
            row.update({
                'pred_visual_heights_1': visual_heights[i, 0],
                'pred_visual_heights_2': visual_heights[i, 1],
                'gt_visual_heights_1': gt_visible[i, 0],
                'gt_visual_heights_2': gt_visible[i, 1],
            })
        rows.append(row)
    return rows


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
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

    out = {}
    for interval, suffix in labels.items():
        group = df[df['gt_ttc_bins'] == interval]
        if len(group) == 0:
            out[f'MiD{suffix}'] = np.nan
            out[f'FR{suffix}'] = np.nan
            out[f'RTE{suffix}'] = np.nan
            continue
        valid = group[~group['_failed']]
        out[f'MiD{suffix}'] = group['MiD'].mean()
        out[f'FR{suffix}'] = group['_failed'].mean() * 100.0
        out[f'RTE{suffix}'] = valid['RTE'].mean()

    valid_all = df[~df['_failed']]
    out['mean_MiD'] = df['MiD'].mean()
    out['paper_MiD_overall'] = sum(
        out[f'MiD{suffix}'] * weight
        for suffix, weight in PAPER_MID_WEIGHTS.items()
    )
    out['mean_RTE'] = valid_all['RTE'].mean()
    out['num_samples'] = len(df)
    out['failed_rate'] = df['_failed'].mean() * 100.0 if len(df) else np.nan
    return pd.DataFrame([out])


def print_summary(summary: pd.DataFrame) -> None:
    row = summary.iloc[0]
    print('MiD/RTE by TTC range: c=(0,3], s=(3,6], l=(6,10], n=(-10,0]')
    print(
        'MiD: '
        f"c={row['MiDc']:.1f} ({row['FRc']:.1f}%), "
        f"s={row['MiDs']:.1f} ({row['FRs']:.1f}%), "
        f"l={row['MiDl']:.1f} ({row['FRl']:.1f}%), "
        f"n={row['MiDn']:.1f} ({row['FRn']:.1f}%)"
    )
    print(
        'RTE: '
        f"c={row['RTEc']:.1f}, s={row['RTEs']:.1f}, "
        f"l={row['RTEl']:.1f}, n={row['RTEn']:.1f}"
    )
    print(
        f"paper_MiD_overall={row['paper_MiD_overall']:.1f}, "
        f"mean_MiD={row['mean_MiD']:.1f}"
    )
    print(f"samples={int(row['num_samples'])}, failed_rate={row['failed_rate']:.1f}%")
