#!/usr/bin/env bash
set -Eeuo pipefail

echo "== Driver / CUDA =="
nvidia-smi
echo
echo "== GPU topology (expected: GPU0 <-> GPU1 is NV2) =="
nvidia-smi topo -m
echo
echo "== NVLink state and per-link bandwidth =="
nvidia-smi nvlink -s
echo
echo "== Python runtime =="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print("GPU{}: {}; CC={}.{}; VRAM={:.2f} GiB".format(
        index, props.name, props.major, props.minor, props.total_memory / 1024**3))
PY

