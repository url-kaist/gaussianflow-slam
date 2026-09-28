"""Compute final ATE from traj_full.txt vs gt_path using evo (matches demo.py monocular eval)."""
import sys
from evo.tools import file_interface
from evo.core import sync, metrics, trajectory

def main():
    if len(sys.argv) < 3:
        print("Usage: compute_ate.py <traj_full.txt> <gt.txt>")
        sys.exit(1)
    est = file_interface.read_tum_trajectory_file(sys.argv[1])
    gt = file_interface.read_tum_trajectory_file(sys.argv[2])
    gt_a, est_a = sync.associate_trajectories(gt, est, max_diff=0.01)
    est_aligned = trajectory.align_trajectory(est_a, gt_a, correct_scale=True)
    ate = metrics.APE(metrics.PoseRelation.translation_part)
    ate.process_data((gt_a, est_aligned))
    s = ate.get_all_statistics()
    print(f"{s['rmse']:.6f}")

if __name__ == "__main__":
    main()
