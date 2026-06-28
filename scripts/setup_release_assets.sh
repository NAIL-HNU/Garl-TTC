#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
EAP_REPO="${EAP_REPO:-NAIL-HNU/eAP-dataset}"
GARLTTC_REPO="${GARLTTC_REPO:-NAIL-HNU/GarlTTC-dataset}"
MODEL_REPO="${MODEL_REPO:-NAIL-HNU/GarlTTC-model}"
EAP_DIR="${EAP_DIR:-$DATA_DIR/eAP-dataset}"
GARLTTC_DIR="${GARLTTC_DIR:-$DATA_DIR/GarlTTC-dataset}"
MODEL_DIR="${MODEL_DIR:-$DATA_DIR/GarlTTC-model}"
HF_HOME="${HF_HOME:-$DATA_DIR/.hf_cache}"
SKIP_UV_SYNC="${SKIP_UV_SYNC:-0}"

export HF_HOME
export ROOT
export DATA_DIR
export EAP_REPO
export GARLTTC_REPO
export MODEL_REPO
export EAP_DIR
export GARLTTC_DIR
export MODEL_DIR

cd "$ROOT"
mkdir -p "$DATA_DIR" "$EAP_DIR" "$GARLTTC_DIR" "$MODEL_DIR" checkpoints outputs

if ! command -v uv >/dev/null 2>&1; then
  python -m pip install uv
fi

if [[ "$SKIP_UV_SYNC" != "1" ]]; then
  uv sync
fi

uv run python - <<'PY'
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def download(repo_id: str, repo_type: str, local_dir: Path, allow_patterns=None) -> None:
    print(f"Downloading {repo_type} repo {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
        token=os.environ.get("HF_TOKEN"),
    )


root = Path(os.environ["ROOT"]) if "ROOT" in os.environ else Path.cwd()
eap_dir = Path(os.environ["EAP_DIR"])
garlttc_dir = Path(os.environ["GARLTTC_DIR"])
model_dir = Path(os.environ["MODEL_DIR"])

download(os.environ["EAP_REPO"], "dataset", eap_dir)
download(os.environ["GARLTTC_REPO"], "dataset", garlttc_dir)
download(
    os.environ["MODEL_REPO"],
    "model",
    model_dir,
    allow_patterns=[
        "README.md",
        "model_release_metadata.json",
        "checkpoints/*.pth",
        "configs/**",
        "metrics/**",
    ],
)

ckpt_src = model_dir / "checkpoints"
ckpt_dst = root / "checkpoints"
ckpt_dst.mkdir(exist_ok=True)
if ckpt_src.is_dir():
    for src in sorted(ckpt_src.glob("*.pth")):
        dst = ckpt_dst / src.name
        print(f"Staging checkpoint {src.name} -> {dst}")
        shutil.copy2(src, dst)

required = [
    eap_dir / "data/train.parquet",
    eap_dir / "data/test.parquet",
    garlttc_dir / "data/train.parquet",
    garlttc_dir / "annotations/train.parquet",
    garlttc_dir / "data/test_inputs.parquet",
    ckpt_dst / "paper_ours_full.pth",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("Missing required release asset(s):\n" + "\n".join(missing))

print("\nRelease assets are ready.")
print(f"eAP data root:      {eap_dir}")
print(f"GarlTTC data root:  {garlttc_dir}")
print(f"Checkpoints root:   {ckpt_dst}")
PY
