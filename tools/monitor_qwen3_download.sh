#!/usr/bin/env bash
set -euo pipefail

model_dir="${QWEN3_MOE_MODEL_DIR:-/root/autodl-tmp/models/Qwen3-30B-A3B-FP8}"
log_file="${QWEN3_MOE_MONITOR_LOG:-/root/autodl-tmp/qwen3_download_monitor.log}"
interval_sec="${QWEN3_MOE_MONITOR_INTERVAL_SEC:-900}"

while true; do
  {
    date -u '+[%Y-%m-%d %H:%M:%S UTC]'
    python3 - "$model_dir" <<'PY'
import glob
import json
import os
import sys

model_dir = sys.argv[1]
index_path = os.path.join(model_dir, "model.safetensors.index.json")
with open(index_path) as f:
    total = int(json.load(f)["metadata"]["total_size"])

paths = sorted(glob.glob(os.path.join(model_dir, "model-*.safetensors")))
current = sum(os.path.getsize(path) for path in paths)
print(f"progress {current / 1024**3:.2f} GiB / {total / 1024**3:.2f} GiB ({current / total * 100:.1f}%)")
for path in paths:
    print(f"  {os.path.basename(path)} {os.path.getsize(path) / 1024**3:.2f} GiB")
PY
    worker_count="$(pgrep -fc 'download_one_qwen3_fp8.sh' || true)"
    curl_count="$(pgrep -fc 'curl .*Qwen3-30B-A3B-FP8' || true)"
    echo "workers bash=${worker_count} curl=${curl_count}"
    df -h "$model_dir" | tail -1
    echo
  } >> "$log_file" 2>&1
  sleep "$interval_sec"
done
