# GarlTTC Release

GarlTTC is the release code for RGB-event object time-to-contact estimation.
This repository contains the training, inference, evaluation, dataset conversion
and CodaBench helper code used by the GarlTTC benchmark release.

Public assets are hosted on Hugging Face:

- eAP media dataset: `NAIL-HNU/eAP-dataset`
- GarlTTC annotations: `NAIL-HNU/GarlTTC-dataset`
- GarlTTC checkpoints: `NAIL-HNU/GarlTTC-model`

Project and benchmark pages:

- eAP dataset project page: https://nail-hnu.github.io/eAP_dataset/
- eAP 3D detection benchmark: https://www.codabench.org/competitions/16717/
- GarlTTC benchmark: https://www.codabench.org/competitions/17289/

Download the eAP dataset first. The GarlTTC dataset references eAP RGB/event
media and does not duplicate those files.

## Layout After Setup

Run the setup script below from the repository root. It creates this layout:

```text
GarlTTC_release/
  data/
    eAP-dataset/        # HF NAIL-HNU/eAP-dataset
    GarlTTC-dataset/    # HF NAIL-HNU/GarlTTC-dataset
    GarlTTC-model/      # HF NAIL-HNU/GarlTTC-model snapshot
  checkpoints/
    paper_ours_full.pth
    paper_visual_only_lhr.pth
    paper_event_only_lhr.pth
    ...
  outputs/
```

The default release config is:

```text
configs/garl_ttc_eventdecoder.yaml
```

It is the full RGB+event model config. The final checkpoint is:

```text
checkpoints/paper_ours_full.pth
```

## Install

Python 3.8 is recommended. The project is packaged with `uv`.

```bash
python -m pip install uv
uv sync
```

## Download Data And Checkpoints

One-command setup:

```bash
bash scripts/setup_release_assets.sh
```

If Hugging Face requires authentication, export a token first:

```bash
export HF_TOKEN=hf_xxx
bash scripts/setup_release_assets.sh
```

Useful environment overrides:

```bash
DATA_DIR=/path/to/assets bash scripts/setup_release_assets.sh
SKIP_UV_SYNC=1 bash scripts/setup_release_assets.sh
```

By default, `DATA_DIR` is `./data`.

## Run Inference

Generate a CodaBench JSON submission on the public test split:

```bash
uv run python tools/infer.py \
  --config configs/garl_ttc_eventdecoder.yaml \
  --checkpoint checkpoints/paper_ours_full.pth \
  --data-root data/eAP-dataset \
  --garlttc-annotation-root data/GarlTTC-dataset \
  --output-json outputs/garlttc_test_submission.json
```

The output schema is:

```json
{
  "meta": {"format": "garlttc_prediction_v1"},
  "results": {
    "<sample_token>": {"ttc": 1.0}
  }
}
```

Zip this JSON as root-level `submission.json` before submitting to CodaBench:

```bash
uv run python - <<'PY'
from pathlib import Path
import zipfile

src = Path("outputs/garlttc_test_submission.json")
dst = Path("outputs/garlttc_test_submission.zip")
with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    zf.write(src, "submission.json")
print(dst)
PY
```

## Evaluate With Private Labels

The public HF dataset does not include test TTC ground truth. If you have the
private CodaBench reference labels locally, run:

```bash
uv run python tools/eval.py \
  --config configs/garl_ttc_eventdecoder.yaml \
  --checkpoint checkpoints/paper_ours_full.pth \
  --data-root data/eAP-dataset \
  --garlttc-annotation-root data/GarlTTC-dataset \
  --test-labels-parquet /path/to/private_labels/test/labels.parquet \
  --asset-file configs/splits/test.txt \
  --output-dir outputs/eval_test12
```

Expected paper test12 row:

```text
MiDc/MiDs/MiDl/MiDn: 53.1 / 37.6 / 40.6 / 31.3
FRc/FRs/FRl/FRn:     0.0 / 0.0 / 0.0 / 0.0
num_samples:         6762
```

## Train

Train with the public HF-style annotations:

```bash
uv run python tools/train.py \
  --config configs/garl_ttc_eventdecoder.yaml \
  --data-root data/eAP-dataset \
  --garlttc-annotation-root data/GarlTTC-dataset \
  --output-dir outputs/train_full
```

For a quick loader smoke test:

```bash
uv run python tools/train.py \
  --config configs/garl_ttc_eventdecoder.yaml \
  --data-root data/eAP-dataset \
  --garlttc-annotation-root data/GarlTTC-dataset \
  --epochs 1 \
  --max-batches 1 \
  --batch-size 1 \
  --num-workers 0 \
  --output-dir outputs/train_smoke
```

The full model uses `paper_visual_only_lhr.pth` and `paper_event_only_lhr.pth`
as branch pretraining checkpoints. The setup script downloads them into
`checkpoints/`.

## Ablation Checkpoints

The model repo also contains ablation checkpoints and matching configs:

```text
configs/ablation/*.yaml
checkpoints/paper_*_baseline.pth
checkpoints/paper_*_lhr*.pth
```

Evaluate one ablation by swapping `--config` and `--checkpoint`, for example:

```bash
uv run python tools/eval.py \
  --config configs/ablation/visual_lhr.yaml \
  --checkpoint checkpoints/paper_visual_only_lhr.pth \
  --data-root data/eAP-dataset \
  --garlttc-annotation-root data/GarlTTC-dataset \
  --test-labels-parquet /path/to/private_labels/test/labels.parquet \
  --asset-file configs/splits/test.txt \
  --output-dir outputs/eval_visual_lhr
```

## CodaBench And Dataset Helpers

Build the GarlTTC HF staging tree from local source annotations:

```bash
uv run python -m garl_ttc_benchmark.build_garlttc_dataset \
  --dataset-info configs/dataset_info.json \
  --garlttc-annotation-root dataset/annotations \
  --eap-public-root data/eAP-dataset \
  --output-root outputs/GarlTTC-dataset-staging \
  --overwrite
```

Build and test the CodaBench bundle:

```bash
uv run python -m garl_ttc_benchmark.codabench.build_codabench_bundle \
  --garlttc-output-root outputs/GarlTTC-dataset-staging \
  --output-dir outputs/GarlTTC-codabench \
  --overwrite \
  --run-local-tests
```

Stage the model repository:

```bash
uv run python -m garl_ttc_benchmark.stage_model_repo \
  --output-root outputs/GarlTTC-model-staging \
  --overwrite
```
