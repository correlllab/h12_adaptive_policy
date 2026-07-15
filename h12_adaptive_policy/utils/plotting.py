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


def plot_summary_multiseed(results, payloads, task, out_path):
    """Multi-seed payload sweep: mean ± std error bars across seeds, FAME vs no-FAME.

    ``results = {"FAME": [seeds_list_per_payload], "no-FAME": [...]}``
    where each ``seeds_list_per_payload`` is a list of metric dicts (one per seed,
    same payload). Metric keys used: ``ee_rmse``, ``ee_b_rmse``, ``base_rmse``,
    ``fell``. Red ✕ marks payloads where any seed fell (fraction shown above).
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    styles = {"FAME": (_FAME_COLOR, "o", "-"), "no-FAME": (_NO_FAME_COLOR, "s", "--")}
    panels = [
        (axes[0], "ee_rmse",   r"world tracking error  $e^W_{ee}$  (after control)"),
        (axes[1], "ee_b_rmse", r"base disturbance at hand  $e^W_{base}$  (IK rejects this)"),
        (axes[2], "base_rmse", "pelvis drift (horizontal)"),
    ]
    n_seeds = max(len(results["FAME"][0]), len(results["no-FAME"][0]))
    for ax, key, title in panels:
        for lab, (c, mk, ls) in styles.items():
            means = np.array([np.mean([m[key] * 100 for m in seeds_list])
                              for seeds_list in results[lab]])
            stds = np.array([np.std([m[key] * 100 for m in seeds_list])
                             for seeds_list in results[lab]])
            ax.errorbar(payloads, means, yerr=stds, fmt=mk, ls=ls, color=c, label=lab,
                        capsize=4, lw=1.6, ms=5, elinewidth=1.2)
            ax.fill_between(payloads, means - stds, means + stds, color=c, alpha=0.12)
            # Mark falls (any seed fell at this payload)
            fall_rates = [np.mean([m["fell"] for m in sl]) for sl in results[lab]]
            xf = [(p, fr) for p, fr in zip(payloads, fall_rates) if fr > 0]
            if xf:
                y_top = (means + stds).max()
                for px, fr in xf:
                    ax.scatter([px], [y_top], marker="x", color="red", s=55, zorder=5)
                    ax.annotate(f"{int(round(fr * 100))}%", (px, y_top),
                                xytext=(0, 6), textcoords="offset points",
                                fontsize=8, color="red", ha="center")
        ax.set_xlabel("payload (kg)"); ax.set_ylabel("cm"); ax.set_title(title)
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.suptitle(
        f"{task}: closed-loop world-frame tracking vs payload — FAME shrinks the disturbance "
        f"the IK must reject  (n={n_seeds} seeds, mean ± std)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved figure -> {out_path}")


def plot_adapt(trajs, step_times, task, out_path):
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


def plot_carry_summary_multiseed(results, payloads, out_path):
    """Bimanual-carry payload sweep: pelvis drift + max tilt vs payload per hand,
    FAME vs no-FAME, with mean ± std error bars across seeds.

    ``results = {"FAME": [[m_seed0, m_seed1, ...], ...], "no-FAME": [...]}`` indexed
    by payload. Each metric dict has ``base_rmse`` (m), ``tilt_max`` (deg), ``fell``.
    Red ✕ marks payloads where any seed fell (fraction shown above).
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    styles = {"FAME": (_FAME_COLOR, "o", "-"), "no-FAME": (_NO_FAME_COLOR, "s", "--")}
    # (axis, metric_key, ylabel, scale_to_display_units)
    panels = [
        (axes[0], "base_rmse", "pelvis drift  (cm)", 100.0),
        (axes[1], "tilt_max",  "max tilt  (deg)",    1.0),
    ]
    n_seeds = max(len(results["FAME"][0]), len(results["no-FAME"][0]))
    for ax, key, ylab, scale in panels:
        for lab, (c, mk, ls) in styles.items():
            means = np.array([np.mean([m[key] * scale for m in seeds_list])
                              for seeds_list in results[lab]])
            stds = np.array([np.std([m[key] * scale for m in seeds_list])
                             for seeds_list in results[lab]])
            ax.errorbar(payloads, means, yerr=stds, fmt=mk, ls=ls, color=c, label=lab,
                        capsize=4, lw=1.6, ms=5, elinewidth=1.2)
            ax.fill_between(payloads, means - stds, means + stds, color=c, alpha=0.12)
            # Fall markers
            fall_rates = [np.mean([m["fell"] for m in sl]) for sl in results[lab]]
            xf = [(p, fr) for p, fr in zip(payloads, fall_rates) if fr > 0]
            if xf:
                y_top = (means + stds).max()
                for px, fr in xf:
                    ax.scatter([px], [y_top], marker="x", color="red", s=55, zorder=5)
                    ax.annotate(f"{int(round(fr * 100))}%", (px, y_top),
                                xytext=(0, 6), textcoords="offset points",
                                fontsize=8, color="red", ha="center")
        ax.set_xlabel("payload per hand (kg)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.suptitle(
        f"bimanual_carry: pelvis drift & max tilt vs payload — FAME vs no-FAME  "
        f"(n={n_seeds} seeds, mean ± std)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved figure -> {out_path}")


def plot_carry_compare(trajs, drop_at_s, torso_schedule, out_path):
    """Bimanual-carry comparison: FAME vs no-FAME over the carry + torso-sweep + drop run.

    Stacks four time-series panels:
      1. torso angle (commanded — shared between conditions)
      2. payload (kg) — shows the load ramp and drop event
      3. encoder latent ||z|| — FAME vs no-FAME
      4. pelvis drift (xy) — the headline metric

    ``trajs`` = {"FAME": traj_dict, "no-FAME": traj_dict}. Vertical markers:
      * blue dashed = torso schedule keyframes (turn start, hold, return)
      * red dotted = drop event
    """
    fame, nof = trajs["FAME"], trajs["no-FAME"]
    t = fame["t"]
    drift = lambda tr: np.linalg.norm(tr["bpos"][:, :2] - tr["bpos"][0, :2], axis=1) * 100

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    # Panel 1 — torso angle
    axes[0].plot(t, np.degrees(fame["torso_cmd"]), "k-", lw=2, label="commanded")
    axes[0].plot(t, np.degrees(fame["torso_actual"]), color=_FAME_COLOR, lw=1.2, alpha=0.85,
                 label="FAME actual")
    axes[0].plot(t, np.degrees(nof["torso_actual"]), color=_NO_FAME_COLOR, ls="--", lw=1.2,
                 alpha=0.85, label="no-FAME actual")
    axes[0].set_ylabel("torso yaw (deg)")
    torso_angles_deg = np.degrees([float(k[1]) for k in torso_schedule])
    sweep_amp_deg = float(torso_angles_deg.max() - torso_angles_deg.min())
    axes[0].set_title(
        f"bimanual_carry: torso {sweep_amp_deg:.0f}° sweep + drop — FAME vs no-FAME"
    )
    axes[0].legend(fontsize=9, loc="upper left")

    # Panel 2 — force per hand (Newtons). Clipping to the encoder's trained envelope
    # still happens inside build_et_mujoco; we just don't annotate the limit here.
    force_N = np.asarray(fame["load"]) * 9.81
    axes[1].plot(t, force_N, "k-", lw=2)
    axes[1].set_ylabel("force per hand (N)")

    # Panel 3 — encoder ||z||
    axes[2].plot(t, fame["z"], color=_FAME_COLOR, lw=1.7, label="FAME (force encoder)")
    axes[2].plot(t, nof["z"], color=_NO_FAME_COLOR, ls="--", lw=1.7, label="no-FAME (blind)")
    axes[2].set_ylabel(r"encoder latent $\|z\|$")
    axes[2].legend(fontsize=9, loc="upper left")

    # Panel 4 — pelvis drift (headline)
    fame_drift = drift(fame); nof_drift = drift(nof)
    axes[3].plot(t, fame_drift, color=_FAME_COLOR, lw=2,
                 label=f"FAME (rmse {np.sqrt(np.mean(fame_drift ** 2)):.2f} cm)")
    axes[3].plot(t, nof_drift, color=_NO_FAME_COLOR, ls="--", lw=2,
                 label=f"no-FAME (rmse {np.sqrt(np.mean(nof_drift ** 2)):.2f} cm)")
    axes[3].set_ylabel("pelvis drift  (cm)")
    axes[3].set_xlabel("t (s)")
    axes[3].legend(fontsize=9, loc="upper left")

    # Vertical guides: torso schedule keyframes (skip the first 0.0) + drop event
    torso_t = [float(k[0]) for k in torso_schedule[1:]]
    for ax in axes:
        for tt in torso_t:
            ax.axvline(tt, color="navy", ls="--", lw=0.8, alpha=0.4)
        ax.axvline(drop_at_s, color="red", ls=":", lw=1.2, alpha=0.7)
        ax.grid(alpha=0.3)
    axes[0].text(drop_at_s, axes[0].get_ylim()[1], "  drop",
                 fontsize=9, color="red", va="top")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved figure -> {out_path}")


def plot_adapt_single(traj, step_times, task, out_path, cond_label="FAME"):
    """Single-condition variant of ``plot_adapt`` for when only one run is available
    (e.g., the DDS demo, where we can't reset the sim between conditions in one
    process). Same 4 panels but only one trace per panel.
    """
    color = _FAME_COLOR if cond_label.upper().startswith("FAME") else _NO_FAME_COLOR
    t = traj["t"]
    werr = np.linalg.norm(traj["world"] - traj["cmd_world"], axis=1) * 100
    drift = np.linalg.norm(traj["bpos"][:, :2] - traj["bpos"][0, :2], axis=1) * 100

    fig, axes = plt.subplots(4, 1, figsize=(12, 10.5), sharex=True)

    axes[0].plot(t, traj["load"], "k-", lw=2)
    axes[0].set_ylabel("payload (kg)")
    axes[0].set_title(f"{task}: payload changed ON THE FLY while moving — {cond_label}")

    axes[1].plot(t, traj["z"], color=color, lw=1.7, label=cond_label)
    axes[1].set_ylabel(r"encoder latent $\|z\|$")
    axes[1].legend(fontsize=9, loc="upper left")

    for ax, y, ylab in [(axes[2], werr, "world EE error (cm)"),
                        (axes[3], drift, "pelvis drift (cm)")]:
        ax.plot(t, y, color=color, lw=1.7, label=cond_label)
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
