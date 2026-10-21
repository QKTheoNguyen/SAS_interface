#!/usr/bin/env bash
# One-time install: model repos, one uv environment per component, checkpoints.
# Safe to re-run; finished steps are fast no-ops.
#
#   ./setup.sh                 # everything
#   ./setup.sh --no-prefetch   # skip checkpoint downloads
set -euo pipefail
cd "$(dirname "$0")"

PREFETCH=1
[[ "${1:-}" == "--no-prefetch" ]] && PREFETCH=0

command -v uv >/dev/null || { echo "uv not found: https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1; }

echo "==> Model repositories (git submodules)"
git submodule update --init --recursive

echo "==> Gateway environment"
uv sync --project gateway

WORKERS=$(uv run --project gateway python -m sas_gateway local-workers | awk '{print $1}')
for worker in $WORKERS; do
  echo "==> Worker environment: $worker"
  uv sync --project "workers/$worker" || {
    echo "Failed to install workers/$worker. If a package failed to build (e.g. pyworld for the metrics worker)," >&2
    echo "make a C/C++ compiler available first. On Triton: module load triton/2024.1-gcc gcc/12.3.0" >&2
    exit 1
  }
  if [[ $PREFETCH == 1 ]]; then
    echo "==> Downloading checkpoints for: $worker"
    uv run --project "workers/$worker" python -m sas_worker prefetch --worker "$worker"
  fi
done

echo
echo "Setup done. Start with ./run.sh (or sbatch scripts/triton_run.sbatch on Triton)."
