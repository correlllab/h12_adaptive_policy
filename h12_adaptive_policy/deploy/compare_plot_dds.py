"""
Offline FAME-vs-no-FAME comparison plot from two .npz traj dumps.

The headline metric is **pelvis drift** over time — FAME's job is to shrink it,
which translates directly to how much disturbance the upper-body IK has to reject.

Each .npz is produced by `manip_ik_demo_dds.py --save_traj <path>`. Pass one FAME
and one no-FAME traj.

Usage:
  # 1. Run FAME (after sim is up; press SPACE in viewer once robot stabilizes)
  python h12_adaptive_policy/deploy/manip_ik_demo_dds.py --task right_hand_manip \
      --save_traj /tmp/run_fame.npz
  # 2. Restart the sim, then run no-FAME
  python h12_adaptive_policy/deploy/manip_ik_demo_dds.py --task right_hand_manip \
      --no_encode --save_traj /tmp/run_no_fame.npz
  # 3. Compare
  python h12_adaptive_policy/deploy/compare_plot_dds.py \
      --fame /tmp/run_fame.npz --no_fame /tmp/run_no_fame.npz \
      --out simulation_exp/figures/compare_pelvis_drift.png
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_TRAJ_KEYS = ("t", "world", "cmd_world", "bpos", "bquat", "load", "z")
_FAME_COLOR = "#2c7fb8"
_NO_FAME_COLOR = "#d95f0e"


def load_traj_npz(path):
    z = np.load(path, allow_pickle=False)
    traj = {k: z[k] for k in _TRAJ_KEYS if k in z.files}
    meta = {
        "cond":      str(z["cond"])      if "cond" in z.files else "?",
        "task":      str(z["task"])      if "task" in z.files else "?",
        "payload":   float(z["payload_kg"]) if "payload_kg" in z.files else float("nan"),
        "adapt":     bool(z["adapt"])    if "adapt" in z.files else False,
        "step_times": z["step_times"]    if "step_times" in z.files else np.array([]),
        "ee_rmse":   float(z["ee_rmse"]) if "ee_rmse" in z.files else float("nan"),
        "ee_b_rmse": float(z["ee_b_rmse"]) if "ee_b_rmse" in z.files else float("nan"),
        "base_rmse": float(z["base_rmse"]) if "base_rmse" in z.files else float("nan"),
        "tilt_max":  float(z["tilt_max"]) if "tilt_max" in z.files else float("nan"),
        "fell":      int(z["fell"])      if "fell" in z.files else 0,
    }
    return traj, meta


def _pelvis_drift_cm(traj):
    """||pelvis_xy(t) − pelvis_xy(0)||  in cm."""
    return np.linalg.norm(traj["bpos"][:, :2] - traj["bpos"][0, :2], axis=1) * 100


def plot_compare(fame_traj, fame_meta, nof_traj, nof_meta, out_path):
    """Single-panel pelvis-drift comparison, FAME vs no-FAME."""
    t_fame = fame_traj["t"]
    t_nof = nof_traj["t"]
    drift_fame = _pelvis_drift_cm(fame_traj)
    drift_nof = _pelvis_drift_cm(nof_traj)

    fig, ax = plt.subplots(figsize=(11, 4.5))

    ax.plot(t_fame, drift_fame, color=_FAME_COLOR, lw=2,
            label=f"FAME (rmse {fame_meta['base_rmse'] * 100:.1f} cm)")
    ax.plot(t_nof, drift_nof, color=_NO_FAME_COLOR, ls="--", lw=2,
            label=f"no-FAME (rmse {nof_meta['base_rmse'] * 100:.1f} cm)")
    ax.set_ylabel("pelvis drift  $\\|p_{xy}(t) - p_{xy}(0)\\|$  (cm)")
    ax.set_xlabel("t (s)")
    ax.set_title(
        f"{fame_meta['task']}: pelvis drift — FAME vs no-FAME"
        + (f"  (payload {fame_meta['payload']:.1f} kg)" if not fame_meta["adapt"]
           else "  (on-the-fly payload)")
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="upper left")
    if fame_meta["fell"] or nof_meta["fell"]:
        notes = []
        if fame_meta["fell"]: notes.append("FAME fell")
        if nof_meta["fell"]: notes.append("no-FAME fell")
        ax.text(0.99, 0.97, " · ".join(notes), transform=ax.transAxes,
                ha="right", va="top", color="red", fontsize=10)

    # Vertical guides at payload step times (--adapt only)
    steps = fame_meta["step_times"]
    if len(steps):
        for s in steps:
            ax.axvline(float(s), color="purple", ls=":", lw=1.0, alpha=0.5)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved figure -> {out_path}")


def main():
    p = argparse.ArgumentParser(description="FAME vs no-FAME pelvis-drift comparison")
    p.add_argument("--fame",    required=True, help="Path to FAME traj .npz")
    p.add_argument("--no_fame", required=True, help="Path to no-FAME traj .npz")
    p.add_argument("--out",     required=True, help="Output PNG path")
    args = p.parse_args()

    fame_traj, fame_meta = load_traj_npz(args.fame)
    nof_traj, nof_meta = load_traj_npz(args.no_fame)

    if fame_meta["task"] != nof_meta["task"]:
        print(f"WARN: tasks differ ({fame_meta['task']!r} vs {nof_meta['task']!r}); "
              "plot will use FAME's.", file=sys.stderr)

    # Print side-by-side task summary
    print("=== task summary ===")
    for lab, m in [("FAME", fame_meta), ("no-FAME", nof_meta)]:
        print(f"  {lab:8s} pelvis drift={m['base_rmse'] * 100:5.2f} cm   "
              f"e^W_ee rmse={m['ee_rmse'] * 100:5.2f} cm   "
              f"e^W_base={m['ee_b_rmse'] * 100:5.2f} cm   "
              f"tilt_max={m['tilt_max']:5.1f} deg   fell={m['fell']}")
    delta = (nof_meta['base_rmse'] - fame_meta['base_rmse']) * 100
    print(f"  Δ(pelvis drift) = no-FAME − FAME = {delta:+.2f} cm  "
          f"({'FAME wins' if delta > 0 else 'no-FAME wins'})")

    plot_compare(fame_traj, fame_meta, nof_traj, nof_meta, args.out)


if __name__ == "__main__":
    main()
