#!/usr/bin/env python3
import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def resolve_record_path(name):
    path = Path(name).expanduser()
    if path.is_absolute():
        return path
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    return DATA_DIR / path.name


def main():
    parser = argparse.ArgumentParser(description="Plot recorded DDS force estimates")
    parser.add_argument("--load", required=True, help="Record filename under data/; .npz is optional")
    parser.add_argument("--save", default=None, help="Optional output image path")
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np

    record_path = resolve_record_path(args.load)
    data = np.load(record_path)
    left_force = data["left_estimated_force"]
    right_force = data["right_estimated_force"]
    x = data["force_time"] if "force_time" in data else np.arange(left_force.shape[0])

    left_mag = np.linalg.norm(left_force, axis=1)
    right_mag = np.linalg.norm(right_force, axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    labels = ["X", "Y", "Z"]
    for idx, label in enumerate(labels):
        axes[0].plot(x, left_force[:, idx], label=f"left {label}")
    axes[0].set_ylabel("Force (N)")
    axes[0].set_title("Left Estimated Force")
    axes[0].legend(ncol=3)
    axes[0].grid(True, alpha=0.3)

    for idx, label in enumerate(labels):
        axes[1].plot(x, right_force[:, idx], label=f"right {label}")
    axes[1].set_ylabel("Force (N)")
    axes[1].set_title("Right Estimated Force")
    axes[1].legend(ncol=3)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, left_mag, label="left |F|")
    axes[2].plot(x, right_mag, label="right |F|")
    axes[2].set_xlabel("Time (s)" if "force_time" in data else "Sample")
    axes[2].set_ylabel("Magnitude (N)")
    axes[2].set_title("Estimated End-Effector Force Magnitude")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    if args.save:
        output_path = Path(args.save).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        print(f"saved plot -> {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
