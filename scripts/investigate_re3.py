#!/usr/bin/env python3
"""
Interactive investigation script for re3_real_encode_tr1 disaster episode.

Usage:
    python scripts/investigate_re3.py [--data DATA_DIR] [--plot] [--pinocchio]

The --pinocchio flag requires the pinocchio environment:
    /home/humanoid/Programs/h12_ros2_controller/.venv/bin/python scripts/investigate_re3.py --pinocchio

Without --pinocchio, all analysis except wrench reconstruction is available.
"""
import argparse
import os
import sys
import numpy as np

DATA_DIR_DEFAULT = "data/real/re3_real_encode_tr1"
URDF_PATH = "/home/humanoid/ws_ctrl/src/h12_ros2_controller/assets/h1_2/h1_2_handless.urdf"

BODY_JOINTS = [
    "l_hip_yaw",    "l_hip_pitch",  "l_hip_roll",   "l_knee",
    "l_ank_pitch",  "l_ank_roll",
    "r_hip_yaw",    "r_hip_pitch",  "r_hip_roll",   "r_knee",
    "r_ank_pitch",  "r_ank_roll",
    "torso",
    "L_sh_pitch",   "L_sh_roll",    "L_sh_yaw",     "L_elbow",
    "L_wr_roll",    "L_wr_pitch",   "L_wr_yaw",
    "R_sh_pitch",   "R_sh_roll",    "R_sh_yaw",     "R_elbow",
    "R_wr_roll",    "R_wr_pitch",   "R_wr_yaw",
]

LEG_IDX   = list(range(0, 12))
TORSO_IDX = 12
LEFT_ARM_IDX  = list(range(13, 20))
RIGHT_ARM_IDX = list(range(20, 27))
ARM_IDX = LEFT_ARM_IDX + RIGHT_ARM_IDX

# ─────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────

def load_data(data_dir):
    time_arr   = np.load(os.path.join(data_dir, "time.npy"))
    tau        = np.load(os.path.join(data_dir, "tau.npy"))
    qpos       = np.load(os.path.join(data_dir, "qpos.npy"))
    dq         = np.load(os.path.join(data_dir, "dq.npy"))
    target_dof = np.load(os.path.join(data_dir, "target_dof.npy"))
    return time_arr, tau, qpos, dq, target_dof


# ─────────────────────────────────────────────────────────────────
# Pinocchio wrench reconstruction
# ─────────────────────────────────────────────────────────────────

def build_pinocchio_model(urdf):
    import pinocchio as pin
    model, _, _ = pin.buildModelsFromUrdf(
        urdf, package_dirs=os.path.dirname(urdf),
        root_joint=pin.JointModelFreeFlyer()
    )
    data = model.createData()
    return model, data

def compute_all_wrenches(model, data, qpos, tau):
    """
    Compute Pinocchio wrist wrenches for every timestep.
    Returns left_wrenches, right_wrenches  (N_steps x 6)
    """
    import pinocchio as pin

    left_id  = model.getFrameId("left_wrist_yaw_link")
    right_id = model.getFrameId("right_wrist_yaw_link")

    def _full_q(q27):
        fq = np.zeros(model.nq); fq[3:7] = [0, 0, 0, 1]; fq[7:] = q27; return fq

    def _grav(q27):
        return pin.rnea(model, data, _full_q(q27), np.zeros(model.nv), np.zeros(model.nv))[6:]

    def _jac(q27, fid):
        fq = _full_q(q27)
        pin.forwardKinematics(model, data, fq)
        pin.updateFramePlacements(model, data)
        J = pin.computeFrameJacobian(
            model, data, fq, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return J[:, 6:]

    def _wrench(q27, tau27, fid):
        tg = _grav(q27)
        J  = _jac(q27, fid)
        return np.linalg.pinv(J.T) @ (tau27 - tg)

    left_wrenches  = np.array([_wrench(qpos[i], tau[i], left_id)  for i in range(len(qpos))])
    right_wrenches = np.array([_wrench(qpos[i], tau[i], right_id) for i in range(len(qpos))])
    return left_wrenches, right_wrenches


def per_joint_contribution(model, data, q27, tau27, frame_name):
    """
    Decompose wrist wrench into per-joint torque contributions.
    Returns dict: joint_name -> (residual, wrench_6d)
    """
    import pinocchio as pin

    fid = model.getFrameId(frame_name)

    fq = np.zeros(model.nq); fq[3:7] = [0, 0, 0, 1]; fq[7:] = q27
    tau_grav = pin.rnea(model, data, fq, np.zeros(model.nv), np.zeros(model.nv))[6:]
    pin.forwardKinematics(model, data, fq)
    pin.updateFramePlacements(model, data)
    J = pin.computeFrameJacobian(model, data, fq, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, 6:]
    JpTinv = np.linalg.pinv(J.T)

    results = {}
    residual = tau27 - tau_grav
    for j, name in enumerate(BODY_JOINTS):
        results[name] = {
            "tau_meas": tau27[j],
            "tau_grav": tau_grav[j],
            "residual": residual[j],
            "wrench_contribution": JpTinv[:, j] * residual[j],
        }
    return results, residual, JpTinv


# ─────────────────────────────────────────────────────────────────
# Text reports
# ─────────────────────────────────────────────────────────────────

def report_overview(time_arr, tau, qpos, dq, target_dof):
    n = len(time_arr)
    dt = np.diff(time_arr)
    print("=" * 70)
    print("EPISODE OVERVIEW")
    print("=" * 70)
    print(f"  Timesteps  : {n}")
    print(f"  Duration   : {time_arr[-1] - time_arr[0]:.3f} s")
    print(f"  Control dt : mean={dt.mean()*1000:.1f} ms  std={dt.std()*1000:.1f} ms  max={dt.max()*1000:.1f} ms")
    print(f"  Time range : {time_arr[0]:.4f} → {time_arr[-1]:.4f} s")


def report_torque_stats(time_arr, tau):
    print()
    print("=" * 70)
    print("JOINT TORQUE STATISTICS  [N·m]")
    print("=" * 70)
    fmt = f"  {'Joint':15s} {'mean':>8} {'std':>8} {'max_abs':>9} {'at step':>8}"
    print(fmt)
    print("  " + "-" * 52)
    for j, name in enumerate(BODY_JOINTS):
        col = tau[:, j]
        peak = np.argmax(np.abs(col))
        print(f"  {name:15s} {col.mean():8.2f} {col.std():8.2f} {np.abs(col).max():9.2f} {peak:8d}")


def report_arm_torques_timeseries(time_arr, tau):
    print()
    print("=" * 70)
    print("ARM TORQUE TIME SERIES (first 15 steps)")
    print("=" * 70)
    header = f"  {'step':>4} {'time':>8} | {'L_sh_p':>8} {'L_sh_r':>8} {'L_elbow':>8} | {'R_sh_p':>8} {'R_sh_r':>8} {'R_elbow':>8} | torso"
    print(header)
    for i in range(min(15, len(time_arr))):
        print(f"  {i:4d} {time_arr[i]:8.4f} | "
              f"{tau[i,13]:8.2f} {tau[i,14]:8.2f} {tau[i,16]:8.2f} | "
              f"{tau[i,20]:8.2f} {tau[i,21]:8.2f} {tau[i,23]:8.2f} | "
              f"{tau[i,12]:7.2f}")


def report_wrench_timeseries(time_arr, left_wrenches, right_wrenches):
    print()
    print("=" * 70)
    print("RECONSTRUCTED PINOCCHIO WRIST WRENCHES  [N]")
    print("=" * 70)
    left_F  = np.linalg.norm(left_wrenches[:, :3], axis=1)
    right_F = np.linalg.norm(right_wrenches[:, :3], axis=1)
    print(f"  LEFT  wrist peak |F| = {left_F.max():.1f} N at step {left_F.argmax()}  t={time_arr[left_F.argmax()]:.4f}s")
    print(f"  RIGHT wrist peak |F| = {right_F.max():.1f} N at step {right_F.argmax()}  t={time_arr[right_F.argmax()]:.4f}s")
    print()
    hdr = f"  {'step':>4} {'time':>8} | {'L_Fx':>8} {'L_Fy':>8} {'L_Fz':>8} {'|L_F|':>8} | {'R_Fx':>8} {'R_Fy':>8} {'R_Fz':>8} {'|R_F|':>8}"
    print(hdr)
    for i in range(len(time_arr)):
        wl = left_wrenches[i]; wr = right_wrenches[i]
        flag = " <== SPIKE" if left_F[i] > 200 else ""
        print(f"  {i:4d} {time_arr[i]:8.4f} | "
              f"{wl[0]:8.1f} {wl[1]:8.1f} {wl[2]:8.1f} {left_F[i]:8.1f} | "
              f"{wr[0]:8.1f} {wr[1]:8.1f} {wr[2]:8.1f} {right_F[i]:8.1f}{flag}")


def report_step_decomposition(model, data, time_arr, tau, qpos, step=8):
    print()
    print("=" * 70)
    print(f"PER-JOINT WRENCH DECOMPOSITION  (step {step}, t={time_arr[step]:.4f}s)")
    print("=" * 70)
    q = qpos[step]; t = tau[step]
    results, residual, JpTinv = per_joint_contribution(model, data, q, t, "left_wrist_yaw_link")

    import pinocchio as pin
    from numpy.linalg import svd
    fq = np.zeros(model.nq); fq[3:7] = [0, 0, 0, 1]; fq[7:] = q
    pin.forwardKinematics(model, data, fq); pin.updateFramePlacements(model, data)
    fid = model.getFrameId("left_wrist_yaw_link")
    J = pin.computeFrameJacobian(model, data, fq, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, 6:]
    _, S, _ = svd(J)
    print(f"  Jacobian singular values: {np.round(S, 4)}")
    print(f"  Condition number: {S[0]/S[-1]:.2f}")

    print()
    print(f"  {'Joint':15s} {'tau_meas':>9} {'tau_grav':>9} {'residual':>9} | {'Fx_contrib':>11} {'Fz_contrib':>11}")
    print("  " + "-" * 72)
    for name, r in results.items():
        contrib = r["wrench_contribution"]
        flag = " <<< ROOT CAUSE" if abs(contrib[2]) > 50 else ""
        print(f"  {name:15s} {r['tau_meas']:9.2f} {r['tau_grav']:9.2f} {r['residual']:9.2f} | "
              f"{contrib[0]:11.2f} {contrib[2]:11.2f}{flag}")

    total = JpTinv @ residual
    print(f"  {'TOTAL':15s}                              | {total[0]:11.2f} {total[2]:11.2f}")


# ─────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────

def make_plots(time_arr, tau, qpos, dq, target_dof,
               left_wrenches=None, right_wrenches=None, save_dir=None):
    import matplotlib
    matplotlib.use("TkAgg" if save_dir is None else "Agg")
    import matplotlib.pyplot as plt

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    def _savefig(fig, name):
        if save_dir:
            path = os.path.join(save_dir, name)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {path}")
        else:
            plt.show()
        plt.close(fig)

    # ── 1. Lower body torques ──────────────────────────────────────
    leg_names = BODY_JOINTS[:12]
    fig, axes = plt.subplots(6, 2, figsize=(16, 18), sharex=True)
    fig.suptitle("Lower Body Joint Torques [N·m]", fontsize=13)
    for j in range(12):
        ax = axes[j // 2, j % 2]
        ax.plot(time_arr, tau[:, j], lw=1.5)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_ylabel(leg_names[j])
        ax.grid(True, alpha=0.4)
    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")
    plt.tight_layout()
    _savefig(fig, "01_lower_body_torques.png")

    # ── 2. Upper body (arm + torso) torques ───────────────────────
    arm_idx = [TORSO_IDX] + LEFT_ARM_IDX + RIGHT_ARM_IDX
    arm_names = [BODY_JOINTS[i] for i in arm_idx]
    fig, axes = plt.subplots(len(arm_idx), 1, figsize=(14, 2.5 * len(arm_idx)), sharex=True)
    fig.suptitle("Upper Body + Arm Joint Torques [N·m]", fontsize=13)
    for k, j in enumerate(arm_idx):
        axes[k].plot(time_arr, tau[:, j], lw=1.5, color="steelblue" if j in LEFT_ARM_IDX else
                     ("darkorange" if j in RIGHT_ARM_IDX else "purple"))
        axes[k].axhline(0, color="gray", lw=0.5, ls="--")
        axes[k].set_ylabel(arm_names[k])
        axes[k].grid(True, alpha=0.4)
    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    _savefig(fig, "02_upper_body_arm_torques.png")

    # ── 3. Arm qpos vs target ─────────────────────────────────────
    fig, axes = plt.subplots(7, 2, figsize=(16, 20), sharex=True)
    fig.suptitle("Arm Joint Position: Measured vs Target [rad]", fontsize=13)
    for k in range(7):
        for side, (base_idx, arm_label) in enumerate([(13, "L"), (20, "R")]):
            ax = axes[k, side]
            j = base_idx + k
            ax.plot(time_arr, qpos[:, j], label="measured", lw=1.5)
            ax.plot(time_arr, target_dof[:, j], label="target", lw=1.5, ls="--")
            ax.set_ylabel(f"{arm_label} {BODY_JOINTS[j].split('_',1)[-1]}")
            ax.grid(True, alpha=0.4)
            if k == 0:
                ax.legend(fontsize=8)
    for side in range(2):
        axes[-1, side].set_xlabel("Time [s]")
    plt.tight_layout()
    _savefig(fig, "03_arm_qpos_vs_target.png")

    # ── 4. Lower body qpos vs target (first 12 joints) ────────────
    fig, axes = plt.subplots(6, 2, figsize=(16, 18), sharex=True)
    fig.suptitle("Lower Body Joint Position: Measured vs Target [rad]", fontsize=13)
    for j in range(12):
        ax = axes[j // 2, j % 2]
        ax.plot(time_arr, qpos[:, j], label="measured", lw=1.5)
        ax.plot(time_arr, target_dof[:, j], label="target", lw=1.5, ls="--")
        ax.set_ylabel(BODY_JOINTS[j])
        ax.grid(True, alpha=0.4)
        if j == 0:
            ax.legend(fontsize=8)
    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")
    plt.tight_layout()
    _savefig(fig, "04_lower_body_qpos_vs_target.png")

    # ── 5. Wrist wrenches ─────────────────────────────────────────
    if left_wrenches is not None:
        left_F  = np.linalg.norm(left_wrenches[:, :3], axis=1)
        right_F = np.linalg.norm(right_wrenches[:, :3], axis=1)

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle("Reconstructed Wrist Force Magnitudes [N]", fontsize=13)

        axes[0].plot(time_arr, left_F,  label="|F| left wrist",  lw=1.5, color="steelblue")
        axes[0].plot(time_arr, right_F, label="|F| right wrist", lw=1.5, color="darkorange")
        axes[0].axhline(200, color="red", ls="--", lw=1, label="200 N threshold")
        axes[0].set_ylabel("|F| [N]")
        axes[0].legend()
        axes[0].grid(True, alpha=0.4)

        axes[1].plot(time_arr, left_wrenches[:, 0], label="Fx", lw=1.2)
        axes[1].plot(time_arr, left_wrenches[:, 1], label="Fy", lw=1.2)
        axes[1].plot(time_arr, left_wrenches[:, 2], label="Fz", lw=1.2)
        axes[1].set_ylabel("Left Wrist F [N]")
        axes[1].legend(); axes[1].grid(True, alpha=0.4)

        axes[2].plot(time_arr, right_wrenches[:, 0], label="Fx", lw=1.2)
        axes[2].plot(time_arr, right_wrenches[:, 1], label="Fy", lw=1.2)
        axes[2].plot(time_arr, right_wrenches[:, 2], label="Fz", lw=1.2)
        axes[2].set_ylabel("Right Wrist F [N]")
        axes[2].set_xlabel("Time [s]")
        axes[2].legend(); axes[2].grid(True, alpha=0.4)

        plt.tight_layout()
        _savefig(fig, "05_wrist_wrenches.png")

        # ── 6. Wrench vs torso residual scatter ─────────────────
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(tau[:, TORSO_IDX], left_F, c=time_arr, cmap="plasma", zorder=3)
        ax.set_xlabel("Torso joint torque τ_est [N·m]")
        ax.set_ylabel("Left wrist |F| [N]")
        ax.set_title("Left wrist force magnitude vs torso torque")
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        _savefig(fig, "06_wrench_vs_torso_tau.png")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Investigate re3 disaster episode")
    parser.add_argument("--data", default=DATA_DIR_DEFAULT, help="Path to episode data directory")
    parser.add_argument("--plot", action="store_true", help="Show/save interactive plots")
    parser.add_argument("--save-plots", metavar="DIR", default=None,
                        help="Save plots to directory instead of displaying")
    parser.add_argument("--pinocchio", action="store_true",
                        help="Compute Pinocchio wrist wrenches (requires pinocchio env)")
    parser.add_argument("--decompose-step", type=int, default=8,
                        help="Timestep for per-joint wrench decomposition (default: 8)")
    args = parser.parse_args()

    time_arr, tau, qpos, dq, target_dof = load_data(args.data)

    report_overview(time_arr, tau, qpos, dq, target_dof)
    report_torque_stats(time_arr, tau)
    report_arm_torques_timeseries(time_arr, tau)

    left_wrenches = right_wrenches = None
    if args.pinocchio:
        sys.path.insert(0, "/home/humanoid/ws_ctrl/src/h12_ros2_controller")
        print()
        print("Building Pinocchio model...")
        model, data = build_pinocchio_model(URDF_PATH)
        print("Computing wrist wrenches for all timesteps...")
        left_wrenches, right_wrenches = compute_all_wrenches(model, data, qpos, tau)
        report_wrench_timeseries(time_arr, left_wrenches, right_wrenches)
        report_step_decomposition(model, data, time_arr, tau, qpos, args.decompose_step)

    if args.plot or args.save_plots:
        make_plots(time_arr, tau, qpos, dq, target_dof,
                   left_wrenches=left_wrenches, right_wrenches=right_wrenches,
                   save_dir=args.save_plots)


if __name__ == "__main__":
    main()
