#!/usr/bin/env python3
"""Run R-PIP-Constraint training for AM, POMO, POMO+PIP (10 epochs, 1000 ep per N)."""
import os, subprocess, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PIP = os.path.join(REPO, "R-PIP-Constraint", "POMO+PIP")
LOG = os.path.join(REPO, "checkpoints", "tsptw")
os.makedirs(LOG, exist_ok=True)
def run(cmd):
    subprocess.run(cmd, cwd=PIP, env={**os.environ, "PYTHONPATH": PIP}, check=True)
for N in (10, 50, 100):
    run([sys.executable, "train.py", "--problem", "STSPTW", "--hardness", "hard", "--problem_size", str(N), "--pomo_size", "1", "--pomo_start", "False", "--epochs", "10", "--train_episodes", "1000", "--train_batch_size", "64", "--log_dir", LOG])
    run([sys.executable, "train.py", "--problem", "STSPTW", "--hardness", "hard", "--problem_size", str(N), "--pomo_size", str(N), "--pomo_start", "True", "--epochs", "10", "--train_episodes", "1000", "--train_batch_size", "64", "--log_dir", LOG])
    run([sys.executable, "train.py", "--problem", "STSPTW", "--hardness", "hard", "--problem_size", str(N), "--pomo_size", str(N), "--pomo_start", "True", "--generate_PI_mask", "--epochs", "10", "--train_episodes", "1000", "--train_batch_size", "64", "--log_dir", LOG])
print("Checkpoints:", LOG)

