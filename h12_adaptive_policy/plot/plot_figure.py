#!/usr/bin/env python3
"""
Generate final summary figure (3 stacked panels) from deploy_real logs.

Expected files under data/real/<run_dir>/:
- qpos.npy
- tau.npy
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

JOINT_INDEX = {name: idx for idx, name in enumerate(JOINT_NAMES)}


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
    tau_path = base / "tau.npy"

    for p in (q_path, tau_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing required file: {p}")

    q = _normalize_joint_data(np.load(q_path), "qpos.npy")
    tau = _normalize_joint_data(np.load(tau_path), "tau.npy")

    if q.shape != tau.shape:
        raise ValueError(f"qpos/tau shapes mismatch: q={q.shape}, tau={tau.shape}")

    return q, tau


def _slice_for_plot(arr: np.ndarray, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    n = arr.shape[0]
    left = min(max(start, 0), n)
    right = min(max(end, 0), n)

    if right <= left:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    frame_idx = np.arange(left, right)
    time_s = (frame_idx - start) / 50.0
    return time_s, arr[left:right]


def plot_final_figure(
    q: np.ndarray,
    tau: np.ndarray,
    start: int,
    end: int,
    output_path: Path,
    no_encode_mode: bool = False,
):
    left_q_color = "#1f77b4"
    right_q_color = "#ff7f0e"
    left_torque_color = "#2ca02c"
    right_torque_color = "#d62728"

    fig_width = 15 * (2.0 / 3.0) if no_encode_mode else 15
    fig, axes_q = plt.subplots(3, 1, figsize=(fig_width, 9), sharex=True)
    axes_tau = [ax.twinx() for ax in axes_q]

    time_full = np.arange(0, end - start) / 50.0
    time_q, q_slice = _slice_for_plot(q, start, end)
    time_tau, tau_slice = _slice_for_plot(tau, start, end)

    l_hip_pitch = JOINT_INDEX["left_hip_pitch_joint"]
    r_hip_pitch = JOINT_INDEX["right_hip_pitch_joint"]
    l_ankle_pitch = JOINT_INDEX["left_ankle_pitch_joint"]
    r_ankle_pitch = JOINT_INDEX["right_ankle_pitch_joint"]
    l_elbow = JOINT_INDEX["left_elbow_joint"]
    r_elbow = JOINT_INDEX["right_elbow_joint"]

    if q_slice.size:
        axes_q[0].plot(
            time_q,
            q_slice[:, l_hip_pitch],
            color=left_q_color,
            label="L Hip Pitch Position",
            linewidth=3.2,
        )
        axes_q[0].plot(
            time_q,
            q_slice[:, r_hip_pitch],
            color=right_q_color,
            label="R Hip Pitch Position",
            linewidth=3.2,
        )
        axes_q[1].plot(
            time_q,
            q_slice[:, l_ankle_pitch],
            color=left_q_color,
            label="L Ankle Pitch Position",
            linewidth=3.2,
        )
        axes_q[1].plot(
            time_q,
            q_slice[:, r_ankle_pitch],
            color=right_q_color,
            label="R Ankle Pitch Position",
            linewidth=3.2,
        )

    if tau_slice.size:
        axes_tau[0].plot(
            time_tau,
            tau_slice[:, l_hip_pitch],
            color=left_torque_color,
            linestyle="--",
            linewidth=2.4,
            label="L Hip Pitch Torque",
        )
        axes_tau[0].plot(
            time_tau,
            tau_slice[:, r_hip_pitch],
            color=right_torque_color,
            linestyle="--",
            linewidth=2.4,
            label="R Hip Pitch Torque",
        )
        axes_tau[1].plot(
            time_tau,
            tau_slice[:, l_ankle_pitch],
            color=left_torque_color,
            linestyle="--",
            linewidth=2.4,
            label="L Ankle Pitch Torque",
        )
        axes_tau[1].plot(
            time_tau,
            tau_slice[:, r_ankle_pitch],
            color=right_torque_color,
            linestyle="--",
            linewidth=2.4,
            label="R Ankle Pitch Torque",
        )
        axes_tau[2].plot(
            time_tau,
            tau_slice[:, l_elbow],
            color=left_torque_color,
            linestyle="--",
            linewidth=2.4,
            label="L Elbow Torque",
        )
        axes_tau[2].plot(
            time_tau,
            tau_slice[:, r_elbow],
            color=right_torque_color,
            linestyle="--",
            linewidth=2.4,
            label="R Elbow Torque",
        )

    axes_q[0].set_ylim(-0.4, 0.3)
    axes_tau[0].set_ylim(-30, 40)

    axes_q[1].set_ylim(-0.45, -0.12)
    axes_tau[1].set_ylim(-10, 45)
    axes_tau[1].set_yticks([-10, 10, 30, 45])

    axes_tau[2].set_ylim(-15, 0)
    axes_tau[2].set_yticks([-15, -10, -5, 0])

    for panel_idx, (ax_q, ax_tau) in enumerate(zip(axes_q, axes_tau)):
        ax_q.grid(True, alpha=0.3)
        ax_q.yaxis.set_ticks_position("left")
        if panel_idx in (0, 1):
            ax_q.tick_params(axis="y", which="both", left=True, right=False, labelsize=14)
        else:
            ax_q.set_yticks([])
            ax_q.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

        ax_tau.yaxis.set_ticks_position("right")
        ax_tau.tick_params(axis="y", which="both", left=False, right=True, labelsize=14)

        if time_full.size > 1:
            ax_q.set_xlim(time_full[0], time_full[-1])

    if no_encode_mode:
        for ax_q, ax_tau in zip(axes_q, axes_tau):
            h_q, l_q = ax_q.get_legend_handles_labels()
            h_tau, l_tau = ax_tau.get_legend_handles_labels()

            handles = h_q + h_tau
            labels = l_q + l_tau

            unique_handles = []
            unique_labels = []
            seen = set()
            for handle, label in zip(handles, labels):
                if label and label not in seen:
                    seen.add(label)
                    unique_handles.append(handle)
                    unique_labels.append(label)

            if unique_handles:
                legend = ax_q.legend(
                    unique_handles,
                    unique_labels,
                    loc="upper left",
                    bbox_to_anchor=(0.52, 0.98),
                    borderaxespad=0.0,
                    fontsize=20,
                    frameon=False,
                )
                for text in legend.get_texts():
                    text.set_ha("left")
                legend._legend_box.align = "left"

    axes_q[-1].set_xlabel("Time (s)", fontsize=18)
    axes_q[-1].tick_params(axis="x", which="both", labelsize=14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate final 3-panel figure from deploy_real logs")
    parser.add_argument(
        "--load",
        type=str,
        required=True,
        help="Folder name under data/real (e.g. re1_real_encode)",
    )
    parser.add_argument("--start", type=int, default=None, help="Start frame (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End frame (exclusive)")
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

    q, tau = load_joint_logs(load_folder)
    sequence_len = q.shape[0]
    no_encode_mode = "no_encode" in load_folder

    start = 0 if args.start is None else args.start
    end = sequence_len if args.end is None else args.end

    if no_encode_mode:
        end = start + int(np.floor((2.0 / 3.0) * (end - start)))

    if args.start is None:
        print("Using default --start: 0")
    if args.end is None:
        print(f"Using default --end: {sequence_len}")

    if start < 0 or end < 0:
        sys.exit("--start and --end must be non-negative")
    if end <= start:
        sys.exit("Invalid frame range: --end must be greater than --start")

    output_path = Path("figure") / "real" / load_folder / "final_figure.png"
    plot_final_figure(q, tau, start, end, output_path, no_encode_mode=no_encode_mode)
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
