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

# Triton GPU nodes have no system C compiler, which torch.compile (Triton kernels) needs.
if ! command -v gcc >/dev/null 2>&1; then
  set +u
  type module >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1
  module load triton/2024.1-gcc gcc/12.3.0 >/dev/null 2>&1 || echo "warning: could not load a gcc module" >&2
  set -u
fi
export CC="${CC:-$(command -v gcc || true)}"
# Keep torch.compile / Triton kernel caches on /scratch so later jobs skip most of the compile time
# (the default is the node-local /tmp, wiped between jobs).
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/scratch/work/$USER/.cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/scratch/work/$USER/.cache/triton}"
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
