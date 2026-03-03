#!/bin/bash
#SBATCH --job-name=tsptw-rl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --output=tsptw_rl_train_%j.out
#SBATCH --error=tsptw_rl_train_%j.err
#
# Optional: request a GPU (uncomment and set partition if your cluster has GPUs)
# #SBATCH --gres=gpu:1
# #SBATCH --partition=your-gpu-partition
#
# Run AM, POMO, POMO+PIP training (10 epochs, 1000 ep per N=10,50,100).
# Submit from repo root:  sbatch models/rl/run_train_tsptw_slurm.sh
# Output/error in submission dir. Checkpoints: checkpoints/tsptw/

set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
REPO="$(pwd)"
if [ ! -f "$REPO/models/rl/run_train_tsptw.py" ]; then
  echo "ERROR: run_train_tsptw.py not found. Submit from repo root: sbatch models/rl/run_train_tsptw_slurm.sh" >&2
  exit 1
fi

module load StdEnv/2023
module load python/3.11.5
module load scipy-stack 2>/dev/null || true

# Install repo + R-PIP-Constraint deps (run on login node if compute has no network)
for REQ in "$REPO/requirements.txt" "$REPO/R-PIP-Constraint/requirements.txt"; do
  if [ -f "$REQ" ]; then
    if ! python -m pip install --user -r "$REQ" --quiet 2>/dev/null; then
      echo "WARN: pip install -r $REQ failed (no network?). Install on login node and re-submit." >&2
    fi
  fi
done

python "$REPO/models/rl/run_train_tsptw.py"
