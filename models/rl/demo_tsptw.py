#!/usr/bin/env python3
"""
Demonstrate TSP-TW pipeline: converter + training/eval commands.

Run from repo root:  python models/rl/demo_tsptw.py

1. Runs TSP-TW -> STSPTW converter on vrp_bench/test_data/tsp_tw_{10,50,100}.pt
2. Prints exact commands to run training and (when eval exists) evaluation.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "vrp_bench"))

def main():
    from vrp_bench.tsptw_to_stsptw import load_tsptw_pt_as_stsptw

    test_dir = os.path.join(REPO, "vrp_bench", "test_data")
    sizes = (10, 50, 100)
    print("=== TSP-TW -> STSPTW converter (load_tsptw_pt_as_stsptw) ===\n")

    for n in sizes:
        path = os.path.join(test_dir, f"tsp_tw_{n}.pt")
        if not os.path.isfile(path):
            print(f"  Skip {path} (not found)")
            continue
        node_xy, service_time, tw_start, tw_end = load_tsptw_pt_as_stsptw(path)
        print(f"  {path}")
        print(f"    shapes: node_xy {tuple(node_xy.shape)}, tw_start {tuple(tw_start.shape)}")
        print(f"    node_xy range: [{node_xy.min():.2f}, {node_xy.max():.2f}] (scaled for STSPTW)")
        print()

    pip_dir = os.path.join(REPO, "R-PIP-Constraint", "POMO+PIP")
    log_dir = os.path.join(REPO, "checkpoints", "tsptw")
    print("=== Training (install deps first: pip install -r R-PIP-Constraint/requirements.txt) ===\n")
    print("  From repo root:")
    print("    python models/rl/run_train_tsptw.py")
    print("  Or from R-PIP-Constraint/POMO+PIP (one size, one algo):")
    print(f"    cd {pip_dir}")
    print('    PYTHONPATH=. python train.py --problem STSPTW --hardness hard --problem_size 10 --pomo_size 1 --pomo_start False --epochs 10 --train_episodes 1000 --log_dir "' + log_dir + '"')
    print()
    print("  Checkpoints will be under:", log_dir)
    print("\nDone.")

if __name__ == "__main__":
    main()
