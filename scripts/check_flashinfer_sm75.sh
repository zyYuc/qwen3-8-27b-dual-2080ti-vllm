#!/usr/bin/env bash
set -Eeuo pipefail

VLLM_PYTHON=python
if [ -f .env ]; then
  source .env
fi

"$VLLM_PYTHON" - <<'PY'
import importlib.metadata as metadata
from pathlib import Path
try:
    version = metadata.version("flashinfer-python")
except metadata.PackageNotFoundError as exc:
    raise SystemExit("flashinfer-python is not installed") from exc
import flashinfer
header = Path(flashinfer.__file__).parent / "data/include/flashinfer/attention/prefill.cuh"
print("flashinfer-python:", version)
print("prefill header:", header)
print("header exists:", header.exists())
if not header.exists():
    raise SystemExit(1)
PY

