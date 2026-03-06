#!/usr/bin/env python3
"""
Plot q, dq, and tau traces for all joints logged by deploy_real.

Expected files under data/real/<run_dir>/:
- qpos.npy
- dq.npy
- tau.npy
- time.npy (optional)
"""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


JOINT_NAMES = [
    "left_hip_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "torso_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def _normalize_joint_data(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")

    n_joints = len(JOINT_NAMES)
    if arr.shape[1] == n_joints:
        return arr
    if arr.shape[0] == n_joints:
        return arr.T

    raise ValueError(
        f"{name} shape {arr.shape} is incompatible with {n_joints} joints "
        f"(expected [T, {n_joints}] or [{n_joints}, T])."
    )


def load_joint_logs(run_dir: str):
    base = Path("data") / "real" / run_dir
    q_path = base / "qpos.npy"
    dq_path = base / "dq.npy"
    tau_path = base / "tau.npy"
    time_path = base / "time.npy"

    for p in (q_path, dq_path, tau_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing required file: {p}")

    q = _normalize_joint_data(np.load(q_path), "qpos.npy")
    dq = _normalize_joint_data(np.load(dq_path), "dq.npy")
    tau = _normalize_joint_data(np.load(tau_path), "tau.npy")

    if not (q.shape == dq.shape == tau.shape):
        raise ValueError(
            f"qpos/dq/tau shapes mismatch: q={q.shape}, dq={dq.shape}, tau={tau.shape}"
        )

    n_samples = q.shape[0]
    if time_path.is_file():
        time_s = np.load(time_path).reshape(-1)
        if time_s.size != n_samples:
            raise ValueError(
                f"time.npy length mismatch: {time_s.size} vs data samples {n_samples}"
            )
    else:
        time_s = np.arange(n_samples, dtype=np.float64)

    return time_s, q, dq, tau


def _base_joint_name(joint_name: str) -> str:
    if joint_name.startswith("left_"):
        return joint_name[len("left_") :]
    if joint_name.startswith("right_"):
        return joint_name[len("right_") :]
    return joint_name


def _joint_groups():
    groups = {}
    order = []

    for idx, name in enumerate(JOINT_NAMES):
        base = _base_joint_name(name)
        if base not in groups:
            groups[base] = {}
            order.append(base)

        if name.startswith("left_"):
            groups[base]["left"] = idx
        elif name.startswith("right_"):
            groups[base]["right"] = idx
        else:
            groups[base]["single"] = idx

    return order, groups


def plot_all_joints(time_s, q, dq, tau, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, joint_name in enumerate(JOINT_NAMES):
        fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)

        axes[0].plot(time_s, q[:, idx], linewidth=1.2)
        axes[0].set_ylabel("q [rad]")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(time_s, dq[:, idx], linewidth=1.2)
        axes[1].set_ylabel("dq [rad/s]")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(time_s, tau[:, idx], linewidth=1.2)
        axes[2].set_ylabel("tau [Nm]")
        axes[2].set_xlabel("Time [s]")
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(joint_name)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        output_path = output_dir / f"{idx:02d}_{joint_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_all_joints_overlay(time_s, q, dq, tau, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_bases, grouped_indices = _joint_groups()

    for base_name in ordered_bases:
        sides = grouped_indices[base_name]
        fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)

        if "single" in sides:
            idx = sides["single"]
            axes[0].plot(time_s, q[:, idx], linewidth=1.2, label=JOINT_NAMES[idx])
            axes[1].plot(time_s, dq[:, idx], linewidth=1.2, label=JOINT_NAMES[idx])
            axes[2].plot(time_s, tau[:, idx], linewidth=1.2, label=JOINT_NAMES[idx])
        else:
            if "left" in sides:
                left_idx = sides["left"]
                axes[0].plot(time_s, q[:, left_idx], linewidth=1.2, label="left")
                axes[1].plot(time_s, dq[:, left_idx], linewidth=1.2, label="left")
                axes[2].plot(time_s, tau[:, left_idx], linewidth=1.2, label="left")

            if "right" in sides:
                right_idx = sides["right"]
                axes[0].plot(time_s, q[:, right_idx], linewidth=1.2, label="right")
                axes[1].plot(time_s, dq[:, right_idx], linewidth=1.2, label="right")
                axes[2].plot(time_s, tau[:, right_idx], linewidth=1.2, label="right")

        axes[0].set_ylabel("q [rad]")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right", fontsize=8)

        axes[1].set_ylabel("dq [rad/s]")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right", fontsize=8)

        axes[2].set_ylabel("tau [Nm]")
        axes[2].set_xlabel("Time [s]")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="upper right", fontsize=8)

        fig.suptitle(base_name)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        output_path = output_dir / f"{base_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot q, dq, tau for all joints from deploy_real logs"
    )
    parser.add_argument(
        "--load",
        type=str,
        required=True,
        help="Folder name under data/real (e.g. re1_real_encode)",
    )
    args = parser.parse_args()

    load_folder = args.load.strip()
    load_folder_path = Path(load_folder)
    if (
        not load_folder
        or load_folder_path.is_absolute()
        or len(load_folder_path.parts) != 1
        or load_folder in {".", ".."}
    ):
        sys.exit("Invalid --load. Please provide a single folder name (no path separators).")

    time_s, q, dq, tau = load_joint_logs(load_folder)

    output_dir_all = Path("figure") / "real" / load_folder / "all_joints"
    output_dir_overlay = Path("figure") / "real" / load_folder / "all_joints_overlay"

    plot_all_joints(time_s, q, dq, tau, output_dir_all)
    plot_all_joints_overlay(time_s, q, dq, tau, output_dir_overlay)

    print(f"Saved joint figures to: {output_dir_all}")
    print(f"Saved overlay figures to: {output_dir_overlay}")


if __name__ == "__main__":
    main()
