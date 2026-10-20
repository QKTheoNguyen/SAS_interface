#!/usr/bin/env bash
# Start the local workers listed in config.yaml, then the gateway (foreground).
# Ctrl+C (or the end of a Slurm job) stops everything.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

while read -r worker host port; do
  [[ -z "$worker" ]] && continue
  log="logs/worker-$worker.log"
  echo "Starting worker $worker on $host:$port (log: $log)"
  uv run --project "workers/$worker" python -m sas_worker serve \
    --worker "$worker" --host "$host" --port "$port" >"$log" 2>&1 &
  pids+=($!)
done < <(uv run --project gateway python -m sas_gateway local-workers)

uv run --project gateway python -m sas_gateway serve &
pids+=($!)
wait -n "${pids[@]}"
echo "A process exited; stopping the rest. Check logs/ for the cause." >&2
