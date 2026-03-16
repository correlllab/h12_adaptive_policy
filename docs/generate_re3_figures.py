#!/usr/bin/env python3
"""
Generate all figures for the RE3 disaster root-cause analysis paper.

Run with the pinocchio venv:
    /home/humanoid/Programs/h12_ros2_controller/.venv/bin/python docs/generate_re3_figures.py

Output: docs/figures/*.pdf
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, "/home/humanoid/ws_ctrl/src/h12_ros2_controller")
import pinocchio as pin

# ─────────────────────────────────────────────────────────────────
# Paths and constants
# ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR  = os.path.join(BASE_DIR, "docs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

URDF = "/home/humanoid/ws_ctrl/src/h12_ros2_controller/assets/h1_2/h1_2_handless.urdf"

BODY_JOINTS = [
    "l_hip_yaw","l_hip_pitch","l_hip_roll","l_knee","l_ank_pitch","l_ank_roll",
    "r_hip_yaw","r_hip_pitch","r_hip_roll","r_knee","r_ank_pitch","r_ank_roll",
    "torso",
    "L_sh_pitch","L_sh_roll","L_sh_yaw","L_elbow","L_wr_roll","L_wr_pitch","L_wr_yaw",
    "R_sh_pitch","R_sh_roll","R_sh_yaw","R_elbow","R_wr_roll","R_wr_pitch","R_wr_yaw",
]

STEPS = 9  # steps 0..8 inclusive

# Colour theme
C_RE3 = "#d62728"   # disaster red
C_RE2 = "#1f77b4"   # good blue
C_TGT = "#ff7f0e"   # target orange (dashed)
ALPHA_SHADE = 0.12

# ─────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────
def load(ep):
    d = os.path.join(BASE_DIR, "data", "real", ep)
    return (np.load(os.path.join(d, "tau.npy")),
            np.load(os.path.join(d, "qpos.npy")),
            np.load(os.path.join(d, "target_dof.npy")),
            np.load(os.path.join(d, "time.npy")),
            np.load(os.path.join(d, "dq.npy")))

re3_tau, re3_q, re3_tgt, re3_t, re3_dq = load("re3_real_encode_tr1")
re2_tau, re2_q, re2_tgt, re2_t, re2_dq = load("re2_real_encode")

steps = np.arange(STEPS)

# ─────────────────────────────────────────────────────────────────
# Pinocchio setup
# ─────────────────────────────────────────────────────────────────
model, _, _ = pin.buildModelsFromUrdf(
    URDF, package_dirs=os.path.dirname(URDF),
    root_joint=pin.JointModelFreeFlyer())
data = model.createData()
lid = model.getFrameId("left_wrist_yaw_link")

def full_q(q27):
    fq = np.zeros(model.nq); fq[3:7] = [0, 0, 0, 1]; fq[7:] = q27
    return fq

def pin_step(q27, tau27):
    fq = full_q(q27)
    tg = pin.rnea(model, data, fq, np.zeros(model.nv), np.zeros(model.nv))[6:]
    pin.forwardKinematics(model, data, fq)
    pin.updateFramePlacements(model, data)
    J = pin.computeFrameJacobian(model, data, fq, lid,
                                 pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, 6:]
    JpTinv = np.linalg.pinv(J.T)
    res = tau27 - tg
    wrench = JpTinv @ res
    _, S, _ = np.linalg.svd(J)
    return dict(wrench=wrench, tg=tg, res=res, J=J, JpTinv=JpTinv,
                S=S, cond=S[0]/S[-1])

print("Computing Pinocchio wrenches for RE3 steps 0-8 ...")
re3_pin = [pin_step(re3_q[s], re3_tau[s]) for s in range(STEPS)]
print("Computing Pinocchio wrenches for RE2 steps 0-8 ...")
re2_pin = [pin_step(re2_q[s], re2_tau[s]) for s in range(STEPS)]

re3_Fz   = np.array([r["wrench"][2] for r in re3_pin])
re3_Fmag = np.array([np.linalg.norm(r["wrench"][:3]) for r in re3_pin])
re2_Fz   = np.array([r["wrench"][2] for r in re2_pin])
re2_Fmag = np.array([np.linalg.norm(r["wrench"][:3]) for r in re2_pin])

re3_torso_res = np.array([r["res"][12] for r in re3_pin])
re2_torso_res = np.array([r["res"][12] for r in re2_pin])

re3_torso_gain = np.array([r["JpTinv"][2, 12] for r in re3_pin])
re2_torso_gain = np.array([r["JpTinv"][2, 12] for r in re2_pin])

# ─────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────
def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

def step_ticks(ax, n=STEPS):
    ax.set_xticks(range(n))
    ax.set_xlim(-0.4, n - 0.6)

def annotate_step8(ax, y, label="step 8", color=C_RE3, offset=(0, 8)):
    ax.annotate(label, xy=(8, y), xytext=(8 + offset[0], y + offset[1]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2),
                color=color, fontsize=7, ha="center")

# ═══════════════════════════════════════════════════════════════════
# Fig 1: Lower-body position errors, steps 0-8 (RE2 vs RE3)
# Focus joints: l_hip_roll (idx 2), r_hip_roll (idx 8), l_knee (idx 3), r_knee (idx 9)
# ═══════════════════════════════════════════════════════════════════
def fig_lower_body_errors():
    joints = [(2, "L Hip Roll"), (8, "R Hip Roll"), (3, "L Knee"), (9, "R Knee"),
              (1, "L Hip Pitch"), (7, "R Hip Pitch")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharey=False)
    axes = axes.flatten()
    for ax, (j, label) in zip(axes, joints):
        re3_err = np.degrees(re3_q[:STEPS, j] - re3_tgt[:STEPS, j])
        re2_err = np.degrees(re2_q[:STEPS, j] - re2_tgt[:STEPS, j])
        ax.plot(steps, re3_err, "o-", color=C_RE3, lw=1.8, ms=4, label="RE3 (disaster)")
        ax.plot(steps, re2_err, "s-", color=C_RE2, lw=1.8, ms=4, label="RE2 (good)")
        ax.axhline(0, color="gray", lw=0.7, ls="--")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Policy step", fontsize=8)
        ax.set_ylabel("Error (deg)", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        step_ticks(ax)
        if j == 2:
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Lower-body joint position errors: $q_{\\mathrm{meas}} - q_{\\mathrm{target}}$ (steps 0–8)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fig, "fig_lower_body_errors.pdf")


# ═══════════════════════════════════════════════════════════════════
# Fig 2: Lower-body measured torques, steps 0-8
# ═══════════════════════════════════════════════════════════════════
def fig_lower_body_torques():
    joints = [(2, "L Hip Roll"), (8, "R Hip Roll"), (3, "L Knee"), (9, "R Knee"),
              (1, "L Hip Pitch"), (7, "R Hip Pitch")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharey=False)
    axes = axes.flatten()
    for ax, (j, label) in zip(axes, joints):
        ax.plot(steps, re3_tau[:STEPS, j], "o-", color=C_RE3, lw=1.8, ms=4, label="RE3 (disaster)")
        ax.plot(steps, re2_tau[:STEPS, j], "s-", color=C_RE2, lw=1.8, ms=4, label="RE2 (good)")
        ax.axhline(0, color="gray", lw=0.7, ls="--")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Policy step", fontsize=8)
        ax.set_ylabel(r"$\tau_{\mathrm{est}}$ (N$\cdot$m)", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        step_ticks(ax)
        if j == 2:
            ax.legend(fontsize=7)
    fig.suptitle(r"Lower-body joint torques $\tau_{\mathrm{est}}$ (steps 0–8)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fig, "fig_lower_body_torques.pdf")


# ═══════════════════════════════════════════════════════════════════
# Fig 3: Commanded targets comparison, steps 0-8
# ═══════════════════════════════════════════════════════════════════
def fig_targets_comparison():
    joints = [(2, "L Hip Roll"), (8, "R Hip Roll"), (3, "L Knee"), (9, "R Knee"),
              (1, "L Hip Pitch"), (7, "R Hip Pitch")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharey=False)
    axes = axes.flatten()
    for ax, (j, label) in zip(axes, joints):
        ax.plot(steps, np.degrees(re3_tgt[:STEPS, j]), "o-", color=C_RE3, lw=1.8, ms=4,
                label="RE3 target")
        ax.plot(steps, np.degrees(re2_tgt[:STEPS, j]), "s-", color=C_RE2, lw=1.8, ms=4,
                label="RE2 target")
        ax.plot(steps, np.degrees(re3_q[:STEPS, j]), "o--", color=C_RE3, lw=1.0, ms=3,
                alpha=0.5, label="RE3 actual")
        ax.plot(steps, np.degrees(re2_q[:STEPS, j]), "s--", color=C_RE2, lw=1.0, ms=3,
                alpha=0.5, label="RE2 actual")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Policy step", fontsize=8)
        ax.set_ylabel("Angle (deg)", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        step_ticks(ax)
        if j == 2:
            ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Policy-commanded targets vs measured positions (steps 0–8)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fig, "fig_targets_comparison.pdf")


# ═══════════════════════════════════════════════════════════════════
# Fig 4: Wrist force cascade — Fz and |F|, steps 0-8
# ═══════════════════════════════════════════════════════════════════
def fig_wrist_force_cascade():
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax = axes[0]
    ax.plot(steps, re3_Fz, "o-", color=C_RE3, lw=2, ms=5, label="RE3 (disaster)")
    ax.plot(steps, re2_Fz, "s-", color=C_RE2, lw=2, ms=5, label="RE2 (good)")
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.axhline(-50, color="purple", lw=1.2, ls=":", label="±50 N clamp (recommended)")
    ax.axhline(50,  color="purple", lw=1.2, ls=":")
    ax.set_ylabel(r"Left wrist $F_z$ (N)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    step_ticks(ax)
    for s, (fz3, fz2) in enumerate(zip(re3_Fz, re2_Fz)):
        if abs(fz3) > 20:
            ax.annotate(f"{fz3:.0f}N", xy=(s, fz3),
                        xytext=(s + 0.1, fz3 - 30 if fz3 < 0 else fz3 + 20),
                        fontsize=6.5, color=C_RE3)

    ax2 = axes[1]
    ax2.plot(steps, re3_Fmag, "o-", color=C_RE3, lw=2, ms=5, label="RE3 |F|")
    ax2.plot(steps, re2_Fmag, "s-", color=C_RE2, lw=2, ms=5, label="RE2 |F|")
    ax2.axhline(50, color="purple", lw=1.2, ls=":", label="50 N clamp")
    ax2.set_ylabel(r"Left wrist $|F|$ (N)", fontsize=9)
    ax2.set_xlabel("Policy step", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    step_ticks(ax2)

    fig.suptitle("Left wrist Pinocchio force estimate: RE3 vs RE2 (steps 0–8)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fig, "fig_wrist_force_cascade.pdf")


# ═══════════════════════════════════════════════════════════════════
# Fig 5: Torso residual → wrist Fz amplification
# ═══════════════════════════════════════════════════════════════════
def fig_torso_cascade():
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    ax0 = axes[0]
    ax0.plot(steps, re3_tau[:STEPS, 12], "o-", color=C_RE3, lw=2, ms=5, label="RE3 torso $\\tau_{\\mathrm{est}}$")
    ax0.plot(steps, re2_tau[:STEPS, 12], "s-", color=C_RE2, lw=2, ms=5, label="RE2 torso $\\tau_{\\mathrm{est}}$")
    ax0.plot(steps, re3_torso_res, "o--", color=C_RE3, lw=1.2, ms=3, alpha=0.6, label="RE3 torso residual")
    ax0.plot(steps, re2_torso_res, "s--", color=C_RE2, lw=1.2, ms=3, alpha=0.6, label="RE2 torso residual")
    ax0.axhline(0, color="gray", lw=0.5, ls="--")
    ax0.set_ylabel(r"Torso $\tau$ (N$\cdot$m)", fontsize=9)
    ax0.legend(fontsize=7.5, ncol=2)
    ax0.grid(True, alpha=0.3)
    step_ticks(ax0)

    ax1 = axes[1]
    ax1.plot(steps, re3_torso_gain, "o-", color=C_RE3, lw=2, ms=5, label=r"RE3: torso$\to F_z$ gain")
    ax1.plot(steps, re2_torso_gain, "s-", color=C_RE2, lw=2, ms=5, label=r"RE2: torso$\to F_z$ gain")
    ax1.set_ylabel(r"$\tilde{J}^+_{[F_z,\,\mathrm{torso}]}$ (N / N$\cdot$m)", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    step_ticks(ax1)

    ax2 = axes[2]
    pred_re3 = re3_torso_gain * re3_torso_res
    pred_re2 = re2_torso_gain * re2_torso_res
    ax2.plot(steps, re3_Fz, "o-", color=C_RE3, lw=2, ms=5, label=r"RE3 total $F_z$")
    ax2.plot(steps, pred_re3, "o--", color=C_RE3, lw=1.3, ms=3, alpha=0.65, label=r"RE3 torso contrib. only")
    ax2.plot(steps, re2_Fz, "s-", color=C_RE2, lw=2, ms=5, label=r"RE2 total $F_z$")
    ax2.plot(steps, pred_re2, "s--", color=C_RE2, lw=1.3, ms=3, alpha=0.65, label=r"RE2 torso contrib. only")
    ax2.axhline(0, color="gray", lw=0.5, ls="--")
    ax2.set_ylabel(r"Left wrist $F_z$ (N)", fontsize=9)
    ax2.set_xlabel("Policy step", fontsize=9)
    ax2.legend(fontsize=7.5, ncol=2)
    ax2.grid(True, alpha=0.3)
    step_ticks(ax2)

    fig.suptitle(r"Torso residual $\to$ wrist $F_z$ amplification chain",
                 fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fig, "fig_torso_cascade.pdf")


# ═══════════════════════════════════════════════════════════════════
# Fig 6: Per-joint Fz contribution at step 8 (RE3)
# ═══════════════════════════════════════════════════════════════════
def fig_per_joint_contribution():
    r = re3_pin[8]
    JpTinv = r["JpTinv"]
    res    = r["res"]
    contribs = JpTinv[2, :] * res  # Fz contributions per joint
    names = [n.replace("_", "\\_") for n in BODY_JOINTS]

    # Sort by absolute contribution descending
    order = np.argsort(np.abs(contribs))[::-1]
    top_n = 15
    order = order[:top_n]
    c_vals = contribs[order]
    c_names = [names[i] for i in order]

    colors = [C_RE3 if v < 0 else "#2ca02c" for v in c_vals]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(range(top_n), c_vals, color=colors, edgecolor="white", height=0.65)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(c_names, fontsize=8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel(r"$F_z$ contribution (N)", fontsize=9)
    ax.set_title(r"Per-joint $F_z$ contribution to left wrist wrench — RE3 step 8"
                 "\n(total: $-533\\,\\mathrm{N}$, torso alone: $-538\\,\\mathrm{N}$)",
                 fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)

    # Label the torso bar
    for i, (idx, val) in enumerate(zip(order, c_vals)):
        if BODY_JOINTS[idx] == "torso":
            ax.annotate(f"torso: {val:.0f} N\n(residual: {res[12]:.1f} N·m × gain {JpTinv[2,12]:.1f})",
                        xy=(val, i), xytext=(val - 10, i + 1.5),
                        fontsize=7, ha="right", color=C_RE3,
                        arrowprops=dict(arrowstyle="-|>", color=C_RE3, lw=1.1))
    plt.tight_layout()
    savefig(fig, "fig_per_joint_contribution.pdf")


# ═══════════════════════════════════════════════════════════════════
# Fig 7: Jacobian singular values — RE2 vs RE3 at step 0 and 8
# ═══════════════════════════════════════════════════════════════════
def fig_singular_values():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, step, title in [
        (axes[0], 0, "Step 0 (episode start)"),
        (axes[1], 8, "Step 8 (Fz = −533 N / 16 N)"),
    ]:
        S3 = re3_pin[step]["S"]
        S2 = re2_pin[step]["S"]
        x  = np.arange(1, len(S3) + 1)
        ax.plot(x, S3, "o-", color=C_RE3, lw=2, ms=6, label=f"RE3  cond={re3_pin[step]['cond']:.1f}")
        ax.plot(x, S2, "s-", color=C_RE2, lw=2, ms=6, label=f"RE2  cond={re2_pin[step]['cond']:.1f}")
        ax.set_yscale("log")
        ax.set_xlabel("Singular value index", fontsize=9)
        ax.set_ylabel(r"$\sigma_i$", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticks(x)
        # Highlight sigma_min
        ax.annotate(f"$\\sigma_6={S3[-1]:.4f}$\n(amplifies ×{1/S3[-1]:.0f})",
                    xy=(6, S3[-1]), xytext=(5.3, S3[-1] * 3),
                    fontsize=7, color=C_RE3,
                    arrowprops=dict(arrowstyle="-|>", color=C_RE3, lw=1.0))
        ax.annotate(f"$\\sigma_6={S2[-1]:.4f}$\n(amplifies ×{1/S2[-1]:.0f})",
                    xy=(6, S2[-1]), xytext=(4.8, S2[-1] * 3),
                    fontsize=7, color=C_RE2,
                    arrowprops=dict(arrowstyle="-|>", color=C_RE2, lw=1.0))

    fig.suptitle(r"Left-wrist Jacobian $J \in \mathbb{R}^{6 \times 27}$ singular values",
                 fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fig, "fig_singular_values.pdf")


# ═══════════════════════════════════════════════════════════════════
# Fig 8: 2D RRR arm — ill-conditioning illustration
# ═══════════════════════════════════════════════════════════════════
def fig_rrr_example():
    """Illustrate Jacobian ill-conditioning for a 2D 3-link (RRR) arm.

    We show two configurations:
      A) Bent arm (θ2 = 90°) — well-conditioned
      B) Near-straight arm (θ2 = 10°) — ill-conditioned

    And compute: if joint-1 has a 5 N·m residual, what end-effector
    force does pinv(J^T) produce?
    """
    L = [0.5, 0.4, 0.3]  # link lengths (m)

    def ee_pos(th):
        th1, th2, th3 = th
        x = (L[0]*np.cos(th1) + L[1]*np.cos(th1+th2) + L[2]*np.cos(th1+th2+th3))
        y = (L[0]*np.sin(th1) + L[1]*np.sin(th1+th2) + L[2]*np.sin(th1+th2+th3))
        return np.array([x, y])

    def jacobian_2d(th):
        th1, th2, th3 = th
        c1  = np.cos(th1);            s1  = np.sin(th1)
        c12 = np.cos(th1+th2);        s12 = np.sin(th1+th2)
        c123= np.cos(th1+th2+th3);    s123= np.sin(th1+th2+th3)
        Jx = np.array([
            -L[0]*s1 - L[1]*s12 - L[2]*s123,
            -L[1]*s12 - L[2]*s123,
            -L[2]*s123,
        ])
        Jy = np.array([
            L[0]*c1 + L[1]*c12 + L[2]*c123,
            L[1]*c12 + L[2]*c123,
            L[2]*c123,
        ])
        return np.vstack([Jx, Jy])  # 2×3

    # Configurations (th1, th2, th3) in radians
    configs = {
        "Bent arm\n($\\theta_2 = 90°$)":
            (np.radians(30), np.radians(90), np.radians(-30)),
        "Near-straight arm\n($\\theta_2 = 10°$)":
            (np.radians(30), np.radians(10), np.radians(-5)),
    }

    residual = np.array([5.0, 0.0, 0.0])  # 5 N·m at joint 1 only

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, (label, th) in zip(axes, configs.items()):
        # Joint positions
        O  = np.array([0.0, 0.0])
        J1 = np.array([L[0]*np.cos(th[0]), L[0]*np.sin(th[0])])
        J2 = J1 + np.array([L[1]*np.cos(th[0]+th[1]), L[1]*np.sin(th[0]+th[1])])
        EE = J2 + np.array([L[2]*np.cos(th[0]+th[1]+th[2]), L[2]*np.sin(th[0]+th[1]+th[2])])

        # Draw links
        xs = [O[0], J1[0], J2[0], EE[0]]
        ys = [O[1], J1[1], J2[1], EE[1]]
        ax.plot(xs, ys, "k-", lw=3, zorder=2)
        for (px, py), mk in [(O, "o"), (J1, "o"), (J2, "o"), (EE, "^")]:
            ax.plot(px, py, mk, color="steelblue", ms=9, zorder=3)

        # Jacobian analysis
        J = jacobian_2d(th)   # 2×3
        JpTinv = np.linalg.pinv(J.T)  # 2×3
        _, S, _ = np.linalg.svd(J)
        F_ee = JpTinv @ residual  # 2-vector
        F_mag = np.linalg.norm(F_ee)

        # Draw force arrow at EE
        scale = 0.25 / max(F_mag, 0.1)
        ax.annotate("", xy=EE + F_ee * scale,
                    xytext=EE,
                    arrowprops=dict(arrowstyle="-|>",
                                   color=C_RE3, lw=2.0, mutation_scale=15))
        ax.text(EE[0] + F_ee[0]*scale*1.05,
                EE[1] + F_ee[1]*scale*1.05 + 0.03,
                f"$F_{{\\mathrm{{EE}}}}={F_mag:.1f}\\,\\mathrm{{N}}$",
                fontsize=8, color=C_RE3, ha="center")

        # Labels
        ax.set_xlim(-0.2, 1.5)
        ax.set_ylim(-0.3, 1.4)
        ax.set_aspect("equal")
        ax.set_title(f"{label}\n$\\sigma_{{\\min}}={S[-1]:.3f}$,  "
                     f"cond$={S[0]/S[-1]:.1f}$,  "
                     f"$|F_{{\\mathrm{{EE}}}}|={F_mag:.1f}$\\,N",
                     fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("$x$ (m)", fontsize=9)
        ax.set_ylabel("$y$ (m)", fontsize=9)

        # Annotate joints
        for pos, name in [(O, "$O$"), (J1, "$J_1$"), (J2, "$J_2$"), (EE, "EE")]:
            ax.annotate(name, xy=pos, xytext=pos + np.array([-0.07, 0.06]),
                        fontsize=8)

        # Print summary
        print(f"  {label.replace(chr(10),' ')}: S={np.round(S,3)}, "
              f"cond={S[0]/S[-1]:.2f}, |F_EE|={F_mag:.2f} N")

    fig.suptitle("2D RRR arm: same 5 N·m joint-1 residual, very different end-effector force",
                 fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, "fig_rrr_example.pdf")


# ═══════════════════════════════════════════════════════════════════
# Fig 9: Torso → wrist Fz gain vs left elbow angle
#        (primary driver: elbow extension opens the arm downward)
#        Also shows wrist height vs elbow to give physical intuition.
# ═══════════════════════════════════════════════════════════════════
def fig_gain_vs_shoulder_angle():
    """
    Sweep L_elbow from 0° (motor zero = forearm at sides, RE1/RE2 pose)
    to 90° (arm fully extended downward, RE3 left arm).
    Hold L_shoulder_pitch at RE3 step-0 value (-16.6°) to isolate elbow.
    A separate panel shows the wrist height (z) to give physical intuition.
    """
    elbow_degs = np.linspace(0, 90, 60)
    gains = []
    conds = []
    wrist_zs = []

    q_base = re3_q[0].copy()
    q_base[13] = np.radians(-16.6)  # fix shoulder pitch at RE3 step-0 value

    lid_l = model.getFrameId("left_wrist_yaw_link")

    for elbow_deg in elbow_degs:
        q = q_base.copy()
        q[16] = np.radians(elbow_deg)   # L_elbow (index 13+3)
        fq = full_q(q)
        pin.forwardKinematics(model, data, fq)
        pin.updateFramePlacements(model, data)
        J = pin.computeFrameJacobian(model, data, fq, lid,
                                     pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, 6:]
        JpTinv = np.linalg.pinv(J.T)
        gains.append(JpTinv[2, 12])
        _, S, _ = np.linalg.svd(J)
        conds.append(S[0] / S[-1])
        wrist_zs.append(data.oMf[lid_l].translation[2])

    fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)

    # Panel 1: wrist height (physical intuition)
    axes[0].plot(elbow_degs, wrist_zs, "k-", lw=2)
    axes[0].axvline(1.67,  color=C_RE2, lw=1.5, ls="--",
                    label=r"RE1/RE2 ($\theta_\text{elb}$=1.7°, physically bent at sides)")
    axes[0].axvline(86.2,  color=C_RE3, lw=1.5, ls="--",
                    label=r"RE3 ($\theta_\text{elb}$=86.2°, arm extended downward)")
    axes[0].axhline(0, color="gray", lw=0.8, ls=":")
    axes[0].set_ylabel("L wrist height $z$ (m)", fontsize=9)
    axes[0].set_title("Wrist descends as elbow extends: RE3 left arm\n"
                      "extends below pelvis ($z < 0$), RE1/RE2 stays above ($z > 0$)",
                      fontsize=8)
    axes[0].legend(fontsize=7.5, loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: torso → Fz gain
    axes[1].plot(elbow_degs, gains, "k-", lw=2)
    axes[1].axvline(1.67, color=C_RE2, lw=1.5, ls="--")
    axes[1].axvline(86.2, color=C_RE3, lw=1.5, ls="--")
    axes[1].set_ylabel(r"Torso $\to$ $F_z$ gain (N / N$\cdot$m)", fontsize=9)
    axes[1].grid(True, alpha=0.3)
    # Annotate the two operating points
    g_re2 = gains[np.argmin(np.abs(elbow_degs - 1.67))]
    g_re3 = gains[np.argmin(np.abs(elbow_degs - 86.2))]
    axes[1].annotate(f"RE1/RE2: {g_re2:.1f}", xy=(1.67, g_re2),
                     xytext=(12, g_re2 + 1.5), fontsize=8, color=C_RE2,
                     arrowprops=dict(arrowstyle="-|>", color=C_RE2, lw=1.0))
    axes[1].annotate(f"RE3: {g_re3:.1f}", xy=(86.2, g_re3),
                     xytext=(70, g_re3 - 3.5), fontsize=8, color=C_RE3,
                     arrowprops=dict(arrowstyle="-|>", color=C_RE3, lw=1.0))

    # Panel 3: condition number
    axes[2].plot(elbow_degs, conds, "k-", lw=2)
    axes[2].axvline(1.67, color=C_RE2, lw=1.5, ls="--")
    axes[2].axvline(86.2, color=C_RE3, lw=1.5, ls="--")
    axes[2].set_ylabel("Jacobian condition number $\\kappa$", fontsize=9)
    axes[2].set_xlabel(r"L elbow motor angle $\theta_\text{elbow}$ (deg)", fontsize=9)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Effect of left elbow extension on wrist Jacobian conditioning\n"
                 r"(L shoulder pitch fixed at $-16.6°$, all other joints at RE3 step-0 values)",
                 fontsize=9, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, "fig_gain_vs_shoulder_angle.pdf")


# ═══════════════════════════════════════════════════════════════════
# Fig 10: Summary cascade panel (all key quantities, steps 0-8)
# ═══════════════════════════════════════════════════════════════════
def fig_cascade_summary():
    fig = plt.figure(figsize=(12, 10))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.38)

    # (0,0) L hip roll error
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(steps, np.degrees(re3_q[:STEPS, 2]-re3_tgt[:STEPS, 2]),
            "o-", color=C_RE3, lw=2, ms=4, label="RE3")
    ax.plot(steps, np.degrees(re2_q[:STEPS, 2]-re2_tgt[:STEPS, 2]),
            "s-", color=C_RE2, lw=2, ms=4, label="RE2")
    ax.set_ylabel("Error (deg)", fontsize=8)
    ax.set_title("L Hip Roll position error", fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); step_ticks(ax)

    # (0,1) R hip roll error
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(steps, np.degrees(re3_q[:STEPS, 8]-re3_tgt[:STEPS, 8]),
            "o-", color=C_RE3, lw=2, ms=4, label="RE3")
    ax.plot(steps, np.degrees(re2_q[:STEPS, 8]-re2_tgt[:STEPS, 8]),
            "s-", color=C_RE2, lw=2, ms=4, label="RE2")
    ax.set_ylabel("Error (deg)", fontsize=8)
    ax.set_title("R Hip Roll position error", fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); step_ticks(ax)

    # (1,0) L knee target
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(steps, np.degrees(re3_tgt[:STEPS, 3]),
            "o-", color=C_RE3, lw=2, ms=4, label="RE3 tgt")
    ax.plot(steps, np.degrees(re2_tgt[:STEPS, 3]),
            "s-", color=C_RE2, lw=2, ms=4, label="RE2 tgt")
    ax.set_ylabel("Angle (deg)", fontsize=8)
    ax.set_title("L Knee commanded target", fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); step_ticks(ax)

    # (1,1) Torso tau
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(steps, re3_tau[:STEPS, 12], "o-", color=C_RE3, lw=2, ms=4,
            label=r"RE3 $\tau_\mathrm{torso}$")
    ax.plot(steps, re2_tau[:STEPS, 12], "s-", color=C_RE2, lw=2, ms=4,
            label=r"RE2 $\tau_\mathrm{torso}$")
    ax.plot(steps, re3_torso_res, "o--", color=C_RE3, lw=1.2, ms=3, alpha=0.6,
            label="RE3 residual")
    ax.set_ylabel(r"$\tau$ (N·m)", fontsize=8)
    ax.set_title("Torso estimated torque", fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); step_ticks(ax)

    # (2,0) Torso → Fz gain
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(steps, re3_torso_gain, "o-", color=C_RE3, lw=2, ms=4, label="RE3 gain")
    ax.plot(steps, re2_torso_gain, "s-", color=C_RE2, lw=2, ms=4, label="RE2 gain")
    ax.set_ylabel(r"Gain (N / N·m)", fontsize=8)
    ax.set_title(r"Torso$\to F_z$ Jacobian gain", fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); step_ticks(ax)

    # (2,1) Wrist Fz
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(steps, re3_Fz, "o-", color=C_RE3, lw=2, ms=4, label="RE3 $F_z$")
    ax.plot(steps, re2_Fz, "s-", color=C_RE2, lw=2, ms=4, label="RE2 $F_z$")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_ylabel("$F_z$ (N)", fontsize=8)
    ax.set_title("Left wrist $F_z$ (Pinocchio)", fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); step_ticks(ax)

    # (3,0) L knee tau
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(steps, re3_tau[:STEPS, 3], "o-", color=C_RE3, lw=2, ms=4, label="RE3")
    ax.plot(steps, re2_tau[:STEPS, 3], "s-", color=C_RE2, lw=2, ms=4, label="RE2")
    ax.set_ylabel(r"$\tau_\mathrm{est}$ (N·m)", fontsize=8)
    ax.set_xlabel("Policy step", fontsize=8)
    ax.set_title("L Knee torque", fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); step_ticks(ax)

    # (3,1) Wrist |F|
    ax = fig.add_subplot(gs[3, 1])
    ax.plot(steps, re3_Fmag, "o-", color=C_RE3, lw=2, ms=4, label=r"RE3 $|F|$")
    ax.plot(steps, re2_Fmag, "s-", color=C_RE2, lw=2, ms=4, label=r"RE2 $|F|$")
    ax.axhline(50, color="purple", lw=1.2, ls=":", label="50 N limit")
    ax.set_ylabel("$|F|$ (N)", fontsize=8)
    ax.set_xlabel("Policy step", fontsize=8)
    ax.set_title("Left wrist force magnitude", fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); step_ticks(ax)

    for ax in fig.get_axes():
        ax.tick_params(labelsize=7)

    fig.suptitle("RE3 vs RE2 cascade summary: steps 0–8", fontsize=11, fontweight="bold")
    savefig(fig, "fig_cascade_summary.pdf")


# ═══════════════════════════════════════════════════════════════════
# Fig 11: Full-run torso tau and wrist forces for RE1 and RE2
#         Shows persistent non-zero torso residual throughout both good runs
# ═══════════════════════════════════════════════════════════════════
def fig_re1re2_full_run():
    """
    Full-run torso tau and Pinocchio wrist |F| for RE1 (2620 steps) and
    RE2 (1357 steps).  Data precomputed (every 5th step) and saved to /tmp/.
    Also shows the hypothetical wrist force if the gain were 15.6 (RE3 level).
    """
    # RE1
    t1   = np.load("/tmp/re1_t.npy")
    Fm1  = np.load("/tmp/re1_Fm.npy")
    tr1  = np.load("/tmp/re1_torso_res.npy")
    tt1_full = np.load("/tmp/re1_torso_tau_full.npy")
    t1_full  = np.load("/tmp/re1_t_full.npy")
    # RE2
    t2   = np.load("/tmp/re2_t.npy")
    Fm2  = np.load("/tmp/re2_Fm.npy")
    tr2  = np.load("/tmp/re2_torso_res.npy")
    tt2_full = np.load("/tmp/re2_torso_tau_full.npy")
    t2_full  = np.load("/tmp/re2_t_full.npy")

    RE3_GAIN = 15.622
    hyp1 = np.abs(tr1) * RE3_GAIN   # lower-bound hypothetical (torso contrib only)
    hyp2 = np.abs(tr2) * RE3_GAIN

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)

    # ── Panel 1: full-run torso tau ──────────────────────────────
    ax = axes[0]
    ax.plot(t1_full, tt1_full, lw=1.0, color=C_RE2, alpha=0.8, label="RE1 torso $\\tau_{\\mathrm{est}}$")
    ax.plot(t2_full + t1_full[-1] + 1.0, tt2_full, lw=1.0, color="teal", alpha=0.8,
            label="RE2 torso $\\tau_{\\mathrm{est}}$")
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    ax.set_ylabel(r"Torso $\tau_{\mathrm{est}}$ (N$\cdot$m)", fontsize=9)
    ax.set_xlabel("Time (s) — RE1 then RE2 (offset by 1 s)", fontsize=8)
    ax.set_title("Persistent non-zero torso torque throughout both successful runs\n"
                 "(caused by imperfect stance: ground-contact forces couple into torso via legs)",
                 fontsize=8.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # ── Panel 2: actual wrist |F| (low because gain=1.5) ────────
    ax = axes[1]
    ax.plot(t1, Fm1, lw=1.0, color=C_RE2, alpha=0.85, label="RE1 wrist $|F|$ (real load applied)")
    ax.plot(t2 + t1[-1] + 1.0, Fm2, lw=1.0, color="teal", alpha=0.85,
            label="RE2 wrist $|F|$ (real load applied)")
    ax.axhline(50, color="purple", lw=1.2, ls=":", label="50 N clamp threshold")
    ax.set_ylabel("Left wrist $|F|$ (N)", fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_title(r"Actual wrist force: max $\approx$ 38 N.  Arm joints compensate real load;"
                 "\ntorso contamination ($\\approx$1.5$\\times$ small residual) is negligible.",
                 fontsize=8.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # ── Panel 3: hypothetical wrist force at RE3 gain (torso contrib only) ─
    ax = axes[2]
    ax.plot(t1, hyp1, lw=1.0, color=C_RE2, alpha=0.85,
            label="RE1 hypothetical (torso contrib, gain=15.6)")
    ax.plot(t2 + t1[-1] + 1.0, hyp2, lw=1.0, color="teal", alpha=0.85,
            label="RE2 hypothetical (torso contrib, gain=15.6)")
    ax.axhline(50, color="purple", lw=1.2, ls=":", label="50 N clamp threshold")
    ax.fill_between(t1, hyp1, 50,
                    where=(hyp1 > 50), alpha=0.18, color=C_RE3, label="above 50 N")
    ax.fill_between(t2 + t1[-1] + 1.0, hyp2, 50,
                    where=(hyp2 > 50), alpha=0.18, color=C_RE3)
    ax.set_ylabel("Hypothetical $|F_z^{\\mathrm{torso}}|$ (N)", fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_title(r"Hypothetical: if RE1/RE2 arms had been in RE3 extended pose (gain $=$ 15.6)"
                 "\nthe same stance-coupling torso residuals would have breached 50 N repeatedly.",
                 fontsize=8.5)
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    fig.suptitle("RE1 and RE2 full runs: torso torque is always non-zero\n"
                 "but arm configuration (gain = 1.5) keeps it harmless",
                 fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fig, "fig_re1re2_full_run.pdf")


# ═══════════════════════════════════════════════════════════════════
# Fig 12: Kinematic junction diagram — torso node schematic
#         (matplotlib only, no tikz dependency)
# ═══════════════════════════════════════════════════════════════════
def fig_kinematic_junction():
    """
    Illustrate the torso as a kinematic junction receiving external
    wrenches from two directions: feet (ground contacts) and wrist
    (intended estimand).  Show that the wrench estimator cannot distinguish
    these sources.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
    import matplotlib.patheffects as pe

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8)
    ax.axis("off")

    def node(cx, cy, label, color, fontsize=9, r=0.55):
        circ = Circle((cx, cy), r, color=color, zorder=3)
        ax.add_patch(circ)
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white", zorder=4)

    def arrow(x0, y0, x1, y1, color="black", label="", lw=1.5, ls="-"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                   linestyle=ls, mutation_scale=14))
        if label:
            mx, my = (x0+x1)/2, (y0+y1)/2
            ax.text(mx+0.12, my, label, fontsize=7.5, color=color,
                    va="center", ha="left",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    def box(cx, cy, w, h, label, color, fontsize=8):
        rect = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                              boxstyle="round,pad=0.1", linewidth=1.2,
                              edgecolor=color, facecolor=color+"22", zorder=3)
        ax.add_patch(rect)
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fontsize, color=color, fontweight="bold", zorder=4,
                multialignment="center")

    # ── nodes ─────────────────────────────────────────────────────
    # Torso (central junction)
    node(5, 4, "TORSO\njoint", "#444444", fontsize=8, r=0.72)

    # Arms
    box(2.1, 5.8, 2.2, 0.9, "Left arm\n(wrist = target)", "#1f77b4", fontsize=8)
    box(7.9, 5.8, 2.2, 0.9, "Right arm", "#aec7e8", fontsize=8)

    # Legs
    box(2.1, 2.2, 2.2, 0.9, "Left leg", "#2ca02c", fontsize=8)
    box(7.9, 2.2, 2.2, 0.9, "Right leg", "#98df8a", fontsize=8)

    # Ground contacts
    box(2.1, 0.7, 2.2, 0.7, "L foot contact\n$F_{\\mathrm{ground}}^L$", "#2ca02c", fontsize=7)
    box(7.9, 0.7, 2.2, 0.7, "R foot contact\n$F_{\\mathrm{ground}}^R$", "#2ca02c", fontsize=7)

    # ── structural arrows ──────────────────────────────────────────
    # Torso ↔ arms
    arrow(3.2, 5.5, 4.3, 4.4, color="#1f77b4", lw=2)
    arrow(6.7, 5.5, 5.7, 4.4, color="#aec7e8", lw=1.5)

    # Torso ↔ legs
    arrow(4.3, 3.6, 3.2, 2.6, color="#2ca02c", lw=2)
    arrow(5.7, 3.6, 6.7, 2.6, color="#98df8a", lw=1.5)

    # Ground → legs
    arrow(2.1, 1.05, 2.1, 1.75, color="#2ca02c", lw=2)
    arrow(7.9, 1.05, 7.9, 1.75, color="#2ca02c", lw=1.5)

    # ── external wrench labels ─────────────────────────────────────
    # Real wrist wrench (small arrow at left arm, during RE1/RE2)
    ax.annotate("", xy=(1.0, 5.8), xytext=(0.1, 6.4),
                arrowprops=dict(arrowstyle="-|>", color="#1f77b4", lw=2.2, mutation_scale=14))
    ax.text(0.05, 6.55, "Real ext.\nwrench\n(RE1/RE2)", fontsize=7, color="#1f77b4",
            ha="center", va="bottom")

    # Spurious reading label at torso
    ax.text(5.0, 4.0, r"$\tau_{\mathrm{torso}}^{\mathrm{est}}$" + "\n≠ 0",
            ha="center", va="center", fontsize=8, color="white", zorder=5, fontweight="bold")

    # ── annotation boxes ──────────────────────────────────────────
    ax.text(5.0, 7.5,
            "Wrench estimator: $\\hat{w} = J^{\\dagger T}(\\tau_{\\mathrm{meas}} - \\tau_{\\mathrm{grav}})$\n"
            "Assumes all $\\tau_{\\mathrm{res}}$ comes from the wrist.  "
            "Ground contacts at feet → torso residual → spurious wrist force.",
            ha="center", va="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fffbe6", ec="orange", lw=1.2))

    ax.text(5.0, 0.25,
            r"$\tau_{\mathrm{torso}}^{\mathrm{res}} = \tau_{\mathrm{from~feet}} + \tau_{\mathrm{inertial}} + \tau_{\mathrm{friction}}$"
            "   (all incorrectly attributed to wrist)",
            ha="center", va="center", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.2", fc="#fff0f0", ec=C_RE3, lw=1.0))

    fig.suptitle("The torso as a kinematic junction: why wrench estimation at the wrist\n"
                 "is contaminated by ground-contact coupling through the legs",
                 fontsize=9.5, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    savefig(fig, "fig_kinematic_junction.pdf")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures...")
    fig_lower_body_errors()
    fig_lower_body_torques()
    fig_targets_comparison()
    fig_wrist_force_cascade()
    fig_torso_cascade()
    fig_per_joint_contribution()
    fig_singular_values()
    fig_rrr_example()
    fig_gain_vs_shoulder_angle()
    fig_cascade_summary()
    fig_re1re2_full_run()
    fig_kinematic_junction()
    print(f"\nAll figures saved to {FIG_DIR}")
