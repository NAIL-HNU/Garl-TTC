from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
import zlib
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


LEADERBOARD_COLUMNS = [
    {'key': 'overall_MiD', 'title': 'Overall MiD', 'index': 0, 'sorting': 'asc'},
    {'key': 'MiDc', 'title': 'MiDc', 'index': 1, 'sorting': 'asc'},
    {'key': 'FRc', 'title': 'FRc', 'index': 2, 'sorting': 'asc'},
    {'key': 'MiDs', 'title': 'MiDs', 'index': 3, 'sorting': 'asc'},
    {'key': 'FRs', 'title': 'FRs', 'index': 4, 'sorting': 'asc'},
    {'key': 'MiDl', 'title': 'MiDl', 'index': 5, 'sorting': 'asc'},
    {'key': 'FRl', 'title': 'FRl', 'index': 6, 'sorting': 'asc'},
    {'key': 'MiDn', 'title': 'MiDn', 'index': 7, 'sorting': 'asc'},
    {'key': 'FRn', 'title': 'FRn', 'index': 8, 'sorting': 'asc'},
    {'key': 'failed_rate', 'title': 'Failed Rate', 'index': 9, 'sorting': 'asc'},
    {'key': 'num_samples', 'title': 'Samples', 'index': 10, 'sorting': 'desc'},
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build the GarlTTC CodaBench benchmark bundle.')
    parser.add_argument('--garlttc-output-root', type=Path, required=True, help='Dataset build root containing public/ and private_labels/.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--hf-repo-id', default='NAIL-HNU/GarlTTC-dataset')
    parser.add_argument('--eap-hf-repo-id', default='NAIL-HNU/eAP-dataset')
    parser.add_argument('--title', default='GarlTTC Benchmark')
    parser.add_argument('--docker-image', default=None)
    parser.add_argument('--contact-email', default='jhanglee@hnu.edu.cn')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--run-local-tests', action='store_true')
    return parser.parse_args(argv)


def reset_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def zip_dir(src_dir: Path, dst_zip: Path) -> None:
    if dst_zip.exists():
        dst_zip.unlink()
    with zipfile.ZipFile(dst_zip, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(src_dir.rglob('*')):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir).as_posix())


def zip_single_file(src_file: Path, dst_zip: Path, arcname: str) -> None:
    if dst_zip.exists():
        dst_zip.unlink()
    with zipfile.ZipFile(dst_zip, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(src_file, arcname)


def write_submission_zip(payload: dict[str, Any], dst_zip: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'submission.json'
        write_json(payload, path)
        zip_single_file(path, dst_zip, 'submission.json')


def copy_scoring_program(bundle_dir: Path) -> Path:
    src_score = Path(__file__).resolve().parent / 'scoring_program' / 'score.py'
    program_dir = bundle_dir / '_scoring_program'
    reset_dir(program_dir, overwrite=True)
    shutil.copy2(src_score, program_dir / 'score.py')
    (program_dir / 'metadata.yaml').write_text(
        'command: python3 /app/program/score.py /app/input/ /app/output/\n',
        encoding='utf-8',
    )
    zip_dir(program_dir, bundle_dir / 'scoring_program.zip')
    shutil.rmtree(program_dir)
    return bundle_dir / 'scoring_program.zip'


def labels_payload(labels: pd.DataFrame) -> dict[str, Any]:
    keep = ['sequence_id', 'sample_token', 'ttc']
    return {
        'meta': {
            'format': 'garlttc_reference_v1',
        },
        'labels': labels[keep].to_dict('records'),
    }


def submission_from_labels(labels: pd.DataFrame) -> dict[str, Any]:
    return {
        'meta': {'format': 'garlttc_prediction_v1'},
        'results': {
            str(row['sample_token']): {'ttc': float(row['ttc'])}
            for row in labels.to_dict('records')
        },
    }


def constant_submission(test_inputs: pd.DataFrame, value: float = 1.0) -> dict[str, Any]:
    return {
        'meta': {'format': 'garlttc_prediction_v1'},
        'results': {
            str(token): {'ttc': value}
            for token in test_inputs['sample_token'].tolist()
        },
    }


def copy_or_make_sample(public_root: Path, test_inputs: pd.DataFrame, dst_zip: Path) -> None:
    sample_path = public_root / 'sample_submission.json'
    if sample_path.exists():
        zip_single_file(sample_path, dst_zip, 'submission.json')
    else:
        write_submission_zip(constant_submission(test_inputs), dst_zip)


def write_reference_zip(labels: pd.DataFrame, dst_zip: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        ref_dir = tmp_dir / 'test'
        ref_dir.mkdir(parents=True)
        write_json(labels_payload(labels), ref_dir / 'labels.json')
        zip_dir(tmp_dir, dst_zip)


def write_pages(bundle_dir: Path, *, hf_repo_id: str, eap_hf_repo_id: str) -> None:
    pages_dir = bundle_dir / 'pages'
    pages_dir.mkdir(parents=True, exist_ok=True)
    (pages_dir / 'overview.md').write_text(
        f"""# GarlTTC Benchmark

The GarlTTC Benchmark evaluates object Time-to-Contact (TTC) estimation for
autonomous driving with synchronized RGB and event camera observations. It is
designed to study event-enhanced visual representation learning for object-level
motion perception under challenging real-world illumination and motion
conditions.

Official dataset page: https://nail-hnu.github.io/eAP_dataset/

Official paper: https://arxiv.org/abs/2603.16303

## Organizers

<div align="center">

<p>
  <img src="https://nail-hnu.github.io/eAP_dataset/assets/logo/NAIL.ico" alt="NAIL Lab" height="96"/><br/>
  <strong>Neuromorphic Automation and Intelligence Lab (NAIL)</strong>
</p>

</div>

## Dataset Introduction

The eAP dataset provides hardware-synchronized RGB, event camera, LiDAR and
GNSS-IMU data for event-enhanced autonomous perception. The paper demonstrates
how eAP supports multiple autonomous perception tasks, including 3D vehicle
detection and object TTC estimation through deep representation learning.

The GarlTTC benchmark focuses on object TTC estimation. Participants predict a
scalar TTC value for each annotated object-centered RGB-event temporal window.
The benchmark follows the paper setting for evaluating geometry-aware
representation learning, where object TTC is related to visual object-height
changes and event-enhanced features help improve perception under challenging
illumination. The released Garl-TTC model is designed for fast object TTC
estimation and is reported in the paper as operating at 200 FPS.

Public benchmark inputs are distributed through two Hugging Face repositories:

https://huggingface.co/datasets/{eap_hf_repo_id}

https://huggingface.co/datasets/{hf_repo_id}

Test TTC ground truth is private and is used only by the CodaBench scoring
program.

## Task

Participants submit one TTC prediction for every public test sample. Each test
sample is keyed by `sample_token`. The prediction value is a scalar TTC in
seconds.

## Metrics

The leaderboard reports Motion-in-Depth error (MiD) and Failure Ratio (FR) over
four TTC ranges:

- `c`: crucial positive TTC, `(0, 3]` seconds
- `s`: small positive TTC, `(3, 6]` seconds
- `l`: large positive TTC, `(6, 10]` seconds
- `n`: negative TTC, `(-10, 0]` seconds

The official leaderboard ranking uses `overall_MiD` in ascending order. It is
computed as:

`overall_MiD = 0.5 * MiDc + 0.3 * MiDs + 0.1 * MiDl + 0.1 * MiDn`

where the weights correspond to `c`, `s`, `l`, and `n`, respectively. Lower
MiD and FR are better.
""",
        encoding='utf-8',
    )
    (pages_dir / 'data.md').write_text(
        f"""# Data

Download public benchmark inputs from Hugging Face:

1. Download https://huggingface.co/datasets/{eap_hf_repo_id} first. It contains
   the RGB/event media used by this benchmark.
2. Download https://huggingface.co/datasets/{hf_repo_id} next. It contains
   GarlTTC train annotations and public test input rows, but it does not
   duplicate RGB/event media.

The public GarlTTC package exposes the 12 test sequences listed in
`data/test_inputs.parquet`; test labels are private. The public package
contains:

- `README.md`
- `data/train.parquet`
- `annotations/train.parquet`
- `data/test_inputs.parquet`
- `splits/train.txt`
- `splits/test.txt`
- `sample_submission.json`

The public package does not contain test TTC labels, private reference JSON or
pkl test annotations. Test labels are used only by the CodaBench scoring
program.
""",
        encoding='utf-8',
    )
    (pages_dir / 'submission.md').write_text(
        """# Submission

Upload a result zip containing exactly one `submission.json` at the root.
CSV submissions are not accepted. Empty submissions, missing sample tokens, or
extra sample tokens fail scoring.

```json
{
  "meta": {"format": "garlttc_prediction_v1"},
  "results": {
    "<sample_token>": {"ttc": 1.0}
  }
}
```

Each key in `results` must match one `sample_token` from the public test input
split. The `ttc` value is the predicted object TTC in seconds.
""",
        encoding='utf-8',
    )
    (pages_dir / 'team.md').write_text(
        """# Team Authors

## Authors

- Jinghang Li*, Ph.D. Student, Hunan University
- Shichao Li*, Senior Research Engineer, ByteDance
- Qing Lian, Researcher, Zhuoyu Technology
- Peiliang Li, Lead, E2E Self-Driving and Next-Gen Algorithms, Zhuoyu Technology
- Xiaozhi Chen, Director of AI Research, Zhuoyu Technology
- Yi Zhou+, Professor, Hunan University

* Equal contribution. + Corresponding author.

## Affiliations

- Neuromorphic Automation and Intelligence Lab (NAIL), Hunan University
- Zhuoyu Technology

## Official Links

- eAP dataset website: https://nail-hnu.github.io/eAP_dataset/
- Paper: https://arxiv.org/abs/2603.16303
""",
        encoding='utf-8',
    )
    (pages_dir / 'terms.md').write_text(
        f"""# Terms

Use the data for benchmark research and follow the dataset licenses in the
Hugging Face repositories:

- https://huggingface.co/datasets/{eap_hf_repo_id}
- https://huggingface.co/datasets/{hf_repo_id}

Do not redistribute private reference data or attempt to recover hidden test
labels.
""",
        encoding='utf-8',
    )


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xFFFFFFFF)


def write_competition_image(bundle_dir: Path) -> str:
    images_dir = bundle_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    path = images_dir / 'garlttc_logo.png'
    width, height = 900, 360
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            band = 1 if (x // 90 + y // 90) % 2 == 0 else 0
            accent = 1 if abs((x - 2 * y) % 220) < 10 else 0
            r = 18 + band * 18 + accent * 110
            g = 38 + band * 24 + accent * 70
            b = 54 + band * 20 + accent * 28
            row.extend((min(r, 255), min(g, 255), min(b, 255), 255))
        rows.append(bytes(row))
    png = (
        b'\x89PNG\r\n\x1a\n'
        + _png_chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0))
        + _png_chunk(b'IDAT', zlib.compress(b''.join(rows), level=9))
        + _png_chunk(b'IEND', b'')
    )
    path.write_bytes(png)
    return 'images/garlttc_logo.png'


def write_competition_yaml(
    bundle_dir: Path,
    *,
    title: str,
    hf_repo_id: str,
    docker_image: str | None,
    contact_email: str,
) -> Path:
    competition = {
        'version': 2,
        'title': title,
        'description': 'GarlTTC object time-to-contact benchmark with private test labels.',
        'image': write_competition_image(bundle_dir),
        'terms': 'pages/terms.md',
        'contact_email': contact_email,
        'registration_auto_approve': True,
        'auto_run_submissions': True,
        'can_participants_make_submissions_public': True,
        'forum_enabled': True,
        'show_detailed_results_in_submission_panel': True,
        'show_detailed_results_in_leaderboard': False,
        'fact_sheet': {
            'method_name': {
                'key': 'method_name',
                'title': 'Method Name',
                'type': 'text',
                'selection': '',
                'is_required': 'true',
                'is_on_leaderboard': 'true',
            },
            'uses_event': {
                'key': 'uses_event',
                'title': 'Uses Event',
                'type': 'checkbox',
                'selection': [True, False],
                'is_required': 'true',
                'is_on_leaderboard': 'true',
            },
            'uses_image': {
                'key': 'uses_image',
                'title': 'Uses Image',
                'type': 'checkbox',
                'selection': [True, False],
                'is_required': 'true',
                'is_on_leaderboard': 'true',
            },
        },
        'pages': [
            {'title': 'Overview', 'file': 'pages/overview.md'},
            {'title': 'Data', 'file': 'pages/data.md'},
            {'title': 'Submission', 'file': 'pages/submission.md'},
            {'title': 'Team', 'file': 'pages/team.md'},
            {'title': 'Terms', 'file': 'pages/terms.md'},
        ],
        'phases': [
            {
                'index': 0,
                'name': 'test',
                'description': f'Private test evaluation for {hf_repo_id}',
                'start': '2026-06-28 00:00:00',
                'end': '2030-12-31 23:59:59',
                'tasks': [0],
                'accepts_only_result_submissions': True,
                'max_submissions_per_day': 20,
                'max_submissions': 200,
            }
        ],
        'tasks': [
            {
                'index': 0,
                'name': 'GarlTTC Test',
                'description': 'Score JSON TTC submissions on the private test reference.',
                'scoring_program': 'scoring_program.zip',
                'reference_data': 'reference_data.zip',
            }
        ],
        'solutions': [
            {
                'index': 0,
                'path': 'sample_submission.zip',
                'tasks': [0],
            }
        ],
        'leaderboards': [
            {
                'key': 'main',
                'title': 'Results',
                'columns': LEADERBOARD_COLUMNS,
            }
        ],
    }
    if docker_image:
        competition['docker_image'] = docker_image
    path = bundle_dir / 'competition.yaml'
    path.write_text(yaml.safe_dump(competition, sort_keys=False), encoding='utf-8')
    return path


def build_bundle(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    bundle_dir = output_dir / 'bundle'
    reset_dir(bundle_dir, overwrite=args.overwrite)

    public_root = args.garlttc_output_root / 'public'
    private_root = args.garlttc_output_root / 'private_labels'
    test_inputs_path = public_root / 'data/test_inputs.parquet'
    labels_path = private_root / 'test/labels.parquet'
    if not test_inputs_path.is_file():
        raise FileNotFoundError(test_inputs_path)
    if not labels_path.is_file():
        raise FileNotFoundError(labels_path)

    test_inputs = pd.read_parquet(test_inputs_path)
    labels = pd.read_parquet(labels_path)
    if set(test_inputs['sample_token']) != set(labels['sample_token']):
        raise ValueError('public test_inputs and private labels sample_token sets differ')

    copy_scoring_program(bundle_dir)
    write_reference_zip(labels, bundle_dir / 'reference_data.zip')
    copy_or_make_sample(public_root, test_inputs, bundle_dir / 'sample_submission.zip')
    write_submission_zip(submission_from_labels(labels), bundle_dir / 'perfect_submission.zip')
    write_pages(bundle_dir, hf_repo_id=args.hf_repo_id, eap_hf_repo_id=args.eap_hf_repo_id)
    competition_yaml = write_competition_yaml(
        bundle_dir,
        title=args.title,
        hf_repo_id=args.hf_repo_id,
        docker_image=args.docker_image,
        contact_email=args.contact_email,
    )
    bundle_zip = output_dir / 'garlttc_benchmark_codabench_bundle.zip'
    zip_dir(bundle_dir, bundle_zip)

    metadata = {
        'bundle_dir': str(bundle_dir),
        'bundle_zip': str(bundle_zip),
        'competition_yaml': str(competition_yaml),
        'leaderboard_score_keys': [col['key'] for col in LEADERBOARD_COLUMNS],
        'num_samples': int(len(labels)),
        'artifacts': {
            'scoring_program': str(bundle_dir / 'scoring_program.zip'),
            'reference_data': str(bundle_dir / 'reference_data.zip'),
            'sample_submission': str(bundle_dir / 'sample_submission.zip'),
            'perfect_submission': str(bundle_dir / 'perfect_submission.zip'),
        },
    }
    write_json(metadata, output_dir / 'codabench_bundle_metadata.json')
    return metadata


def run_local_score(scoring_program_zip: Path, reference_zip: Path, submission_zip: Path, output_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        program_dir = tmp_dir / 'program'
        program_dir.mkdir()
        with zipfile.ZipFile(scoring_program_zip, 'r') as zf:
            zf.extractall(program_dir)
        proc = subprocess.run(
            [
                sys.executable,
                str(program_dir / 'score.py'),
                '--submission-zip',
                str(submission_zip),
                '--reference-zip',
                str(reference_zip),
                '--output',
                str(output_dir),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            raise RuntimeError(f'Scoring failed for {submission_zip}:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}')
    return read_json(output_dir / 'scores.json')


def run_local_score_expect_failure(
    scoring_program_zip: Path,
    reference_zip: Path,
    submission_zip: Path,
    output_dir: Path,
    expected_text: str,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        program_dir = tmp_dir / 'program'
        program_dir.mkdir()
        with zipfile.ZipFile(scoring_program_zip, 'r') as zf:
            zf.extractall(program_dir)
        proc = subprocess.run(
            [
                sys.executable,
                str(program_dir / 'score.py'),
                '--submission-zip',
                str(submission_zip),
                '--reference-zip',
                str(reference_zip),
                '--output',
                str(output_dir),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        combined = proc.stdout + proc.stderr
        if proc.returncode == 0:
            raise RuntimeError(f'Expected scoring failure for {submission_zip}, but it succeeded.')
        if expected_text not in combined:
            raise RuntimeError(
                f"Expected {expected_text!r} in scoring failure for {submission_zip}.\n"
                f'STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}'
            )


def assert_local_tests(metadata: dict[str, Any], output_dir: Path) -> None:
    artifacts = metadata['artifacts']
    scoring_zip = Path(artifacts['scoring_program'])
    reference_zip = Path(artifacts['reference_data'])
    sample_zip = Path(artifacts['sample_submission'])
    perfect_zip = Path(artifacts['perfect_submission'])
    tests_dir = output_dir / 'local_scoring_tests'
    reset_dir(tests_dir, overwrite=True)

    sample_scores = run_local_score(scoring_zip, reference_zip, sample_zip, tests_dir / 'sample')
    perfect_scores = run_local_score(scoring_zip, reference_zip, perfect_zip, tests_dir / 'perfect')
    if int(sample_scores['num_samples']) != int(metadata['num_samples']):
        raise RuntimeError(f'Sample submission did not score all rows: {sample_scores}')
    if int(perfect_scores['num_samples']) != int(metadata['num_samples']):
        raise RuntimeError(f'Perfect submission did not score all rows: {perfect_scores}')
    for key in ('MiDc', 'MiDs', 'MiDl', 'MiDn', 'failed_rate', 'overall_MiD'):
        if abs(float(perfect_scores[key])) > 1e-6:
            raise RuntimeError(f'Expected perfect {key}=0, got {perfect_scores[key]}')

    metadata_zip = tests_dir / 'metadata_submission.zip'
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        write_json(read_submission_from_zip(perfect_zip), tmp_dir / 'submission.json')
        (tmp_dir / 'metadata').write_text('codabench metadata placeholder\n', encoding='utf-8')
        zip_dir(tmp_dir, metadata_zip)
    metadata_scores = run_local_score(scoring_zip, reference_zip, metadata_zip, tests_dir / 'metadata')
    if int(metadata_scores['num_samples']) != int(metadata['num_samples']):
        raise RuntimeError(f'Metadata submission did not score all rows: {metadata_scores}')

    bad_format = {
        'meta': {'format': 'wrong_format'},
        'results': {},
    }
    bad_format_zip = tests_dir / 'bad_format_submission.zip'
    write_submission_zip(bad_format, bad_format_zip)
    run_local_score_expect_failure(scoring_zip, reference_zip, bad_format_zip, tests_dir / 'bad_format', 'meta.format')

    empty_submission = {
        'meta': {'format': 'garlttc_prediction_v1'},
        'results': {},
    }
    empty_zip = tests_dir / 'empty_submission.zip'
    write_submission_zip(empty_submission, empty_zip)
    run_local_score_expect_failure(scoring_zip, reference_zip, empty_zip, tests_dir / 'empty', 'sample_token set mismatch')

    bad_token = read_submission_from_zip(perfect_zip)
    first_token = next(iter(bad_token['results']))
    bad_token['results'].pop(first_token)
    bad_token_zip = tests_dir / 'bad_token_submission.zip'
    write_submission_zip(bad_token, bad_token_zip)
    run_local_score_expect_failure(scoring_zip, reference_zip, bad_token_zip, tests_dir / 'bad_token', 'sample_token set mismatch')

    csv_zip = tests_dir / 'csv_submission.zip'
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / 'submission.csv'
        csv_path.write_text('sample_token,ttc\nabc,1.0\n', encoding='utf-8')
        zip_single_file(csv_path, csv_zip, 'submission.csv')
    run_local_score_expect_failure(scoring_zip, reference_zip, csv_zip, tests_dir / 'csv', 'only accepts JSON')


def read_submission_from_zip(zip_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open('submission.json') as fh:
            return json.loads(fh.read().decode('utf-8'))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    metadata = build_bundle(args)
    if args.run_local_tests:
        assert_local_tests(metadata, Path(metadata['bundle_dir']).parent)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
