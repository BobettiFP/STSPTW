#!/bin/bash
#SBATCH --job-name=tsptw-unified
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --output=tsptw_unified_eval_%j.out
#SBATCH --error=tsptw_unified_eval_%j.err
#
# Run unified TSP-TW evaluation (evaluate_unified.py): npz or torch format.
# Submit from repo root: sbatch vrp_bench/run_evaluate_unified_slurm.sh [npz|torch]
# Optional env: RUN=YYYY-MM-DD_HH-MM-SS SIZES="10 50 100" MAX_INSTANCES=20
# Example: SIZES="10 50 100" MAX_INSTANCES=20 sbatch vrp_bench/run_evaluate_unified_slurm.sh npz
# Output/error in submission dir. Results: vrp_bench/eval_results/<run>_<format>/

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
if [ -f main.py ]; then
  : # already in vrp_bench
elif [ -f vrp_bench/main.py ]; then
  cd vrp_bench
else
  echo "ERROR: main.py not found. Submit from repo root: sbatch vrp_bench/run_evaluate_unified_slurm.sh [npz|torch]" >&2
  exit 1
fi

module load StdEnv/2023
module load python/3.11.5
module load scipy-stack 2>/dev/null || true

REQ="../requirements.txt"
if [ ! -f "$REQ" ]; then
  REQ="${SLURM_SUBMIT_DIR:-.}/requirements.txt"
fi
if ! python -m pip install --user -r "$REQ" --quiet 2>/dev/null; then
  echo "ERROR: pip install failed. On the login node run once:" >&2
  echo "  module load StdEnv/2023 python/3.11.5" >&2
  echo "  pip install --user -r $(cd .. 2>/dev/null && pwd)/requirements.txt" >&2
  exit 1
fi

FORMAT="${1:-npz}"
SIZES="${SIZES:-}"
MAX_INSTANCES="${MAX_INSTANCES:-}"

build_args() {
  local args="--format $FORMAT"
  [ -n "$RUN" ] && args="$args --run $RUN"
  [ -n "$SIZES" ] && args="$args --sizes $SIZES"
  [ -n "$MAX_INSTANCES" ] && args="$args --max-instances $MAX_INSTANCES"
  echo "$args"
}

python evaluate_unified.py $(build_args)
