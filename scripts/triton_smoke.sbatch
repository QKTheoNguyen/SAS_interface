#!/usr/bin/env bash
# Start everything on a GPU node, run scripts/smoke_test.py, then exit.
#   sbatch scripts/triton_smoke.sbatch
# Results: logs/sas-smoke-<jobid>.out and logs/smoke/*.wav
#SBATCH --job-name=sas-smoke
#SBATCH --partition=gpu-debug
#SBATCH --gpus=1
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/sas-smoke-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

./run.sh &
RUN_PID=$!
PORT=$(uv run --project gateway python -c "from sas_gateway.config import load_config; print(load_config().port)")
uv run --project gateway python scripts/smoke_test.py --url "http://127.0.0.1:$PORT"
STATUS=$?
nvidia-smi --query-gpu=memory.used --format=csv,noheader || true
kill $RUN_PID
wait $RUN_PID 2>/dev/null
echo "--- worker logs (errors only) ---"
grep -hE "ERROR|Traceback|Error:" logs/worker-*.log | tail -20
exit $STATUS
