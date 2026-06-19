"""Figures for the manip_ik_demo experiments (trajectory, payload sweep, on-the-fly adapt).

Each plot accepts the trajectory dicts produced by ``run_manip``:
    traj = {
        "t":         (N,)    sim time (s),
        "world":     (N, 3)  measured EE in world frame (m),
        "cmd_world": (N, 3)  commanded EE in world frame (m, fixed target),
        "bpos":      (N, 3)  pelvis position (m),
        "bquat":     (N, 4)  pelvis quat (w, x, y, z),
        "load":      (N,)    payload (kg) on the manip hand,
        "z":         (N,)    encoder latent magnitude (||z||),
    }
``trajs_by_cond`` is ``{"FAME": traj, "no-FAME": traj}``.

Uses the matplotlib Agg backend (no display required).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .math_helpers import quat2R


_FAME_COLOR = "#2c7fb8"
_NO_FAME_COLOR = "#d95f0e"


def _base_frame(traj):
    """Express actual & commanded EE in the base frame.

    Actual EE uses the current base pose at each step; the commanded EE is expressed
    in the t=0 (nominal) base frame so the gap shows how the arm DEVIATES from its
    nominal command to absorb base drift.
    """
    p0 = traj["bpos"][0]
    R0 = quat2R(traj["bquat"][0])
    act_b = np.array([quat2R(q).T @ (w - p) for w, p, q in
                      zip(traj["world"], traj["bpos"], traj["bquat"])])
    cmd_b = np.array([R0.T @ (c - p0) for c in traj["cmd_world"]])
    return act_b, cmd_b


def plot_traj(trajs, payload, task, out_path):
    """Commanded vs actual EE position (FAME vs no-FAME), WORLD frame and BASE frame.

    WORLD row: the hand holds the fixed world target; the controller compensates base
    motion (e^W_ee small). BASE row: the arm deviates from its nominal command to
    absorb base drift (larger for no-FAME).
    """
    fame, nof = trajs["FAME"], trajs["no-FAME"]
    t = fame["t"]
    cmd_w = fame["cmd_world"]
    fa_b, cmd_b = _base_frame(fame)
    nf_b, _ = _base_frame(nof)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    names = ["x (forward)", "y (left)", "z (up)"]
    for j in range(3):
        a0, a1 = axes[0, j], axes[1, j]
        a0.plot(t, cmd_w[:, j] * 100, "k-", lw=2, label="commanded")
        a0.plot(t, fame["world"][:, j] * 100, color=_FAME_COLOR, lw=1.6, label="FAME actual")
        a0.plot(t, nof["world"][:, j] * 100, color=_NO_FAME_COLOR, ls="--", lw=1.6, label="no-FAME actual")
        a0.set_title(f"WORLD  {names[j]}"); a0.grid(alpha=.3)
        a1.plot(t, cmd_b[:, j] * 100, "k-", lw=2)
        a1.plot(t, fa_b[:, j] * 100, color=_FAME_COLOR, lw=1.6)
        a1.plot(t, nf_b[:, j] * 100, color=_NO_FAME_COLOR, ls="--", lw=1.6)
        a1.set_title(f"BASE  {names[j]}"); a1.grid(alpha=.3); a1.set_xlabel("t (s)")
    axes[0, 0].set_ylabel("position (cm)"); axes[1, 0].set_ylabel("position (cm)")
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle(
        f"{task}, {payload} kg — fixed WORLD target vs actual hand (closed-loop tracking)\n"
        "WORLD frame: the hand holds the fixed world target — the controller compensates base motion (e^W_ee small).   "
        "BASE frame: the arm DEVIATES from its nominal command to absorb base drift — more for no-FAME",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved figure -> {out_path}")


def plot_summary(results, payloads, task, out_path):
    """Sweep summary: e^W_ee, e^W_base, base drift vs payload, FAME vs no-FAME.

    ``results`` is ``{"FAME": [metrics_dict, ...], "no-FAME": [...]}`` indexed by payload.
    Each metrics_dict has keys ``ee_rmse``, ``ee_b_rmse``, ``base_rmse``, ``fell``.
    Falls are marked with red X at the top of each panel.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    styles = {"FAME": (_FAME_COLOR, "o", "-"), "no-FAME": (_NO_FAME_COLOR, "s", "--")}
    panels = [
        (axes[0], "ee_rmse",   r"world tracking error  $e^W_{ee}$  (after control)"),
        (axes[1], "ee_b_rmse", r"base disturbance at hand  $e^W_{base}$  (IK rejects this)"),
        (axes[2], "base_rmse", "pelvis drift (horizontal)"),
    ]
    for ax, key, title in panels:
        for lab, (c, mk, ls) in styles.items():
            y = [r[key] * 100 for r in results[lab]]
            ax.plot(payloads, y, ls, color=c, marker=mk, label=lab)
            xf = [p for p, r in zip(payloads, results[lab]) if r["fell"]]
            if xf:
                ax.scatter(xf, [max(y)] * len(xf), marker="x", color="red", zorder=5)
        ax.set_xlabel("payload (kg)"); ax.set_ylabel("cm"); ax.set_title(title)
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.suptitle(
        f"{task}: closed-loop world-frame tracking vs payload — FAME shrinks the disturbance the IK must reject",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved figure -> {out_path}")


def plot_adapt(trajs, step_times, task, out_path, trained_limit_kg=3.06):
    """Time series under an ON-THE-FLY changing payload: FAME (force encoder, adapts)
    vs no-FAME (blind). Vertical purple lines mark sudden load changes.

    Panels: payload, encoder latent ||z||, world EE error, pelvis drift.
    """
    fame, nof = trajs["FAME"], trajs["no-FAME"]
    t = fame["t"]
    werr = lambda tr: np.linalg.norm(tr["world"] - tr["cmd_world"], axis=1) * 100
    drift = lambda tr: np.linalg.norm(tr["bpos"][:, :2] - tr["bpos"][0, :2], axis=1) * 100

    fig, axes = plt.subplots(4, 1, figsize=(12, 10.5), sharex=True)

    axes[0].plot(t, fame["load"], "k-", lw=2)
    axes[0].set_ylabel("payload (kg)")
    axes[0].axhline(trained_limit_kg, color="gray", ls=":", lw=1)
    axes[0].text(t[2], trained_limit_kg + 0.04, "30 N trained limit", fontsize=8, color="gray")
    axes[0].set_title(
        f"{task}: payload changed ON THE FLY while moving — FAME adapts (force encoder) vs no-FAME (blind)"
    )

    axes[1].plot(t, fame["z"], color=_FAME_COLOR, lw=1.7, label="FAME (force encoder)")
    axes[1].plot(t, nof["z"], color=_NO_FAME_COLOR, ls="--", lw=1.7, label="no-FAME (blind)")
    axes[1].set_ylabel(r"encoder latent $\|z\|$")
    axes[1].legend(fontsize=9, loc="upper left")

    for ax, fn, ylab in [(axes[2], werr, "world EE error (cm)"), (axes[3], drift, "pelvis drift (cm)")]:
        ax.plot(t, fn(fame), color=_FAME_COLOR, lw=1.7, label="FAME (adapts)")
        ax.plot(t, fn(nof), color=_NO_FAME_COLOR, ls="--", lw=1.7, label="no-FAME (blind)")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=9, loc="upper left")
    axes[3].set_xlabel("t (s)")

    for ax in axes:
        for s in step_times:
            ax.axvline(s, color="purple", ls=":", lw=1.2, alpha=0.6)
        ax.grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved figure -> {out_path}")
