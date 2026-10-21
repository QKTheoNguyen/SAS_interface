#!/usr/bin/env bash
# Run the interface on a Slurm GPU node. Generic template: adapt the #SBATCH lines and the
# module/cache settings below to your cluster (see scripts/triton_run.sbatch for an Aalto Triton example).
# Submit from the repo root:
#   sbatch scripts/run.sbatch
# Then follow the tunnel instructions printed in logs/sas-interface-<jobid>.out.
#
# Command-line flags override these lines, e.g.
#   sbatch --partition=<debug-partition> --time=00:30:00 scripts/run.sbatch
# List partitions and free nodes with:
#   sinfo -o "%P %t %D"
#SBATCH --job-name=sas-interface
#SBATCH --partition=<gpu-partition>
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/sas-interface-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# torch.compile (Triton kernels) needs a C compiler. If your GPU nodes lack one,
# set GCC_MODULES to the module(s) providing gcc, e.g. GCC_MODULES="gcc/12.3.0".
GCC_MODULES="${GCC_MODULES:-}"
if ! command -v gcc >/dev/null 2>&1 && [ -n "$GCC_MODULES" ]; then
  set +u
  type module >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1
  module load $GCC_MODULES >/dev/null 2>&1 || echo "warning: could not load $GCC_MODULES" >&2
  set -u
fi
export CC="${CC:-$(command -v gcc || true)}"
# Keep torch.compile / Triton kernel caches on persistent storage so later jobs skip most of the
# compile time (the default is node-local /tmp, often wiped between jobs). Set CACHE_ROOT to a
# persistent path on your cluster.
CACHE_ROOT="${CACHE_ROOT:-$HOME/.cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"

# Login host used as the SSH jump host for the tunnel.
LOGIN_HOST="${LOGIN_HOST:-<cluster-login-host>}"

NODE=$(hostname -s)
PORT=$(uv run --project gateway python -c "from sas_gateway.config import load_config; print(load_config().port)")
cat <<EOF
=====================================================================
Job $SLURM_JOB_ID on $NODE, GPU: ${CUDA_VISIBLE_DEVICES:-none}
On your laptop, open the tunnel:

  ssh -L $PORT:localhost:$PORT -J $USER@$LOGIN_HOST $USER@$NODE

then browse to http://localhost:$PORT
Stop with: scancel $SLURM_JOB_ID
=====================================================================
EOF
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

exec ./run.sh
