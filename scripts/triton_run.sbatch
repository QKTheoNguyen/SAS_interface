#!/usr/bin/env bash
# Run the interface on a Triton GPU node. Submit from the repo root:
#   sbatch scripts/triton_run.sbatch
# Then follow the tunnel instructions printed in logs/sas-interface-<jobid>.out.
#
# Command-line flags override these lines, e.g.
#   sbatch --partition=gpu-debug --time=00:30:00 scripts/triton_run.sbatch
# Check which partitions have free nodes first:
#   sinfo -p gpu-a100-80g,gpu-h200-141g-short,gpu-h200-71g-ia,gpu-debug -o "%P %t %D"
# (state drain/drng/down = unusable). Alternatives:
#   --partition=gpu-h200-71g-ia --gpus=h200_3g.71gb:1      (H200 slice, 8 h max, single node)
#   --partition=gpu-h200-141g-short --gpus=1               (full H200, 2 days max)
#   --partition=gpu-debug --gpus=1 --time=00:30:00         (A100/V100, 30 min max)
#SBATCH --job-name=sas-interface
#SBATCH --partition=gpu-a100-80g
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/sas-interface-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

NODE=$(hostname -s)
PORT=$(uv run --project gateway python -c "from sas_gateway.config import load_config; print(load_config().port)")
cat <<EOF
=====================================================================
Job $SLURM_JOB_ID on $NODE, GPU: ${CUDA_VISIBLE_DEVICES:-none}
On your laptop, open the tunnel:

  ssh -L $PORT:localhost:$PORT -J $USER@triton.aalto.fi $USER@$NODE

then browse to http://localhost:$PORT
Stop with: scancel $SLURM_JOB_ID
=====================================================================
EOF
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

exec ./run.sh
