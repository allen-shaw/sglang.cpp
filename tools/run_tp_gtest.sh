#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "usage: $0 TP_SIZE TEST_BINARY GTEST_FILTER [extra gtest args...]" >&2
  exit 2
fi

tp_size="$1"
test_binary="$2"
gtest_filter="$3"
shift 3

master_addr="${SGLANG_TP_MASTER_ADDR:-127.0.0.1}"
master_port="${SGLANG_TP_MASTER_PORT:-29500}"
log_dir="${SGLANG_TP_LOG_DIR:-/tmp/sglang_tp_${tp_size}_$(date -u +%Y%m%d_%H%M%S)}"
nccl_id_file="${SGLANG_TP_NCCL_ID_FILE:-$log_dir/nccl_unique_id}"

mkdir -p "$log_dir"

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

for rank in $(seq 0 $((tp_size - 1))); do
  log_file="$log_dir/rank_${rank}.log"
  (
    export WORLD_SIZE="$tp_size"
    export RANK="$rank"
    export LOCAL_RANK="$rank"
    export SGLANG_TP_SIZE="$tp_size"
    export SGLANG_TP_RANK="$rank"
    export SGLANG_TP_LOCAL_RANK="$rank"
    export MASTER_ADDR="$master_addr"
    export MASTER_PORT="$master_port"
    export SGLANG_TP_MASTER_ADDR="$master_addr"
    export SGLANG_TP_MASTER_PORT="$master_port"
    export SGLANG_TP_NCCL_ID_FILE="$nccl_id_file"
    exec "$test_binary" --gtest_filter="$gtest_filter" --gtest_color=no "$@"
  ) >"$log_file" 2>&1 &
  pids+=("$!")
done

status=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  if ! wait "$pid"; then
    echo "rank $i failed; log: $log_dir/rank_${i}.log" >&2
    status=1
  fi
done

if [ "$status" -ne 0 ]; then
  echo "logs: $log_dir" >&2
  exit "$status"
fi

echo "all ranks passed; logs: $log_dir"
