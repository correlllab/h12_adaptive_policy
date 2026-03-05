#!/usr/bin/env python3
"""
Plot knee/ankle torque traces logged by deploy_real.

Expected CSV format (from deploy_real.py):
time,L_knee,L_ankle_pitch,L_ankle_roll,R_knee,R_ankle_pitch,R_ankle_roll
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path("logs/real/knee_ankle_tau.csv")
DEFAULT_OUTPUT = Path("figure/knee_ankle_tau.png")


def load_tau_csv(csv_path: Path):
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=np.float64)
    if data.size == 0:
        raise ValueError(f"Input CSV is empty: {csv_path}")

    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)

    required_cols = [
        "time",
        "L_knee",
        "L_ankle_pitch",
        "L_ankle_roll",
        "R_knee",
        "R_ankle_pitch",
        "R_ankle_roll",
    ]
    missing = [col for col in required_cols if col not in data.dtype.names]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    time = data["time"]
    torque = {name: data[name] for name in required_cols[1:]}
    return time, torque


def plot_knee_ankle_tau(time_s, tau_dict, output_path: Path, title: str):
    joint_names = list(tau_dict.keys())
    n_joints = len(joint_names)

    fig, axes = plt.subplots(n_joints, 1, figsize=(12, 2.5 * n_joints), sharex=True)
    if n_joints == 1:
        axes = [axes]

    for i, joint in enumerate(joint_names):
        axes[i].plot(time_s, tau_dict[joint], linewidth=1.4, label=f"{joint} tau")
        axes[i].set_ylabel("tau [Nm]")
        axes[i].set_title(joint)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="upper right", fontsize=8)

    axes[0].set_title(title)
    axes[-1].set_xlabel("Time [s]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot knee/ankle torques from deploy_real CSV log")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output image path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Knee/Ankle Joint Torque (Measured)",
        help="Figure title",
    )
    args = parser.parse_args()

    time_s, tau_dict = load_tau_csv(args.input)
    plot_knee_ankle_tau(time_s, tau_dict, args.output, args.title)

    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
