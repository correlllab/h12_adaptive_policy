#!/usr/bin/env python3
"""
Interactive explorer for any real-robot episode directory.

Usage:
    python scripts/explore_episode.py --data data/real/re3_real_encode_tr1
    python scripts/explore_episode.py --data data/real/re3_real_encode_tr1 --step 8 --joints torso l_knee r_knee

For pinocchio wrench reconstruction:
    /home/humanoid/Programs/h12_ros2_controller/.venv/bin/python scripts/explore_episode.py \\
        --data data/real/re3_real_encode_tr1 --pin --step 8
"""
import argparse
import os
import sys
import numpy as np

BODY_JOINTS = [
    "l_hip_yaw",   "l_hip_pitch", "l_hip_roll",  "l_knee",
    "l_ank_pitch", "l_ank_roll",
    "r_hip_yaw",   "r_hip_pitch", "r_hip_roll",  "r_knee",
    "r_ank_pitch", "r_ank_roll",
    "torso",
    "L_sh_pitch",  "L_sh_roll",   "L_sh_yaw",    "L_elbow",
    "L_wr_roll",   "L_wr_pitch",  "L_wr_yaw",
    "R_sh_pitch",  "R_sh_roll",   "R_sh_yaw",    "R_elbow",
    "R_wr_roll",   "R_wr_pitch",  "R_wr_yaw",
]

URDF_PATH = "/home/humanoid/ws_ctrl/src/h12_ros2_controller/assets/h1_2/h1_2_handless.urdf"


def load(data_dir):
    d = {
        "time":       np.load(os.path.join(data_dir, "time.npy")),
        "tau":        np.load(os.path.join(data_dir, "tau.npy")),
        "qpos":       np.load(os.path.join(data_dir, "qpos.npy")),
        "dq":         np.load(os.path.join(data_dir, "dq.npy")),
        "target_dof": np.load(os.path.join(data_dir, "target_dof.npy")),
    }
    return d


def summary(d):
    t = d["time"]
    print(f"\n{'─'*60}")
    print(f"  Steps     : {len(t)}")
    print(f"  Duration  : {t[-1]-t[0]:.3f} s   ({t[0]:.3f} → {t[-1]:.3f})")
    dt = np.diff(t) * 1000
    print(f"  dt (ms)   : mean={dt.mean():.1f}  std={dt.std():.1f}  max={dt.max():.1f}")
    print(f"  tau range : {d['tau'].min():.2f} … {d['tau'].max():.2f} N·m")
    print(f"  qpos range: {d['qpos'].min():.4f} … {d['qpos'].max():.4f} rad")


def show_step(d, step):
    t = d["time"]
    print(f"\n{'─'*60}")
    print(f"  Step {step}  /  t = {t[step]:.4f} s  (t+{t[step]-t[0]:.4f} s from start)")
    print(f"  {'Joint':16s}  {'qpos (deg)':>12}  {'dq (deg/s)':>12}  {'tau_est (Nm)':>13}  {'target (deg)':>13}  {'dev (deg)':>10}")
    for j, name in enumerate(BODY_JOINTS):
        q  = np.degrees(d["qpos"][step, j])
        dq = np.degrees(d["dq"][step, j])
        ta = d["tau"][step, j]
        tg = np.degrees(d["target_dof"][step, j])
        dev = q - tg
        flag = " !" if abs(ta) > 80 else ""
        print(f"  {name:16s}  {q:12.2f}  {dq:12.2f}  {ta:13.2f}  {tg:13.2f}  {dev:10.2f}{flag}")


def show_joint(d, joint_names):
    indices = []
    for jname in joint_names:
        matches = [i for i, n in enumerate(BODY_JOINTS) if jname.lower() in n.lower()]
        if not matches:
            print(f"  [warn] '{jname}' not found. Available: {BODY_JOINTS}")
        indices.extend(matches)

    if not indices:
        return

    t = d["time"]
    print(f"\n{'─'*60}")
    for j in indices:
        name = BODY_JOINTS[j]
        tau_col  = d["tau"][:, j]
        qpos_col = np.degrees(d["qpos"][:, j])
        tgt_col  = np.degrees(d["target_dof"][:, j])
        peak_tau = np.argmax(np.abs(tau_col))
        print(f"\n  Joint: {name}  (index {j})")
        print(f"    tau_est : mean={tau_col.mean():.2f}  std={tau_col.std():.2f}  max_abs={np.abs(tau_col).max():.2f} Nm  (peak at step {peak_tau}, t={t[peak_tau]:.4f}s)")
        print(f"    qpos    : mean={qpos_col.mean():.2f} deg  range=[{qpos_col.min():.2f}, {qpos_col.max():.2f}] deg")
        print(f"    target  : mean={tgt_col.mean():.2f} deg")
        print()
        print(f"    {'step':>5}  {'time':>8}  {'tau_est':>10}  {'qpos (deg)':>12}  {'target (deg)':>13}  {'dev':>8}")
        for i in range(len(t)):
            flag = " <MAX" if i == peak_tau else ""
            print(f"    {i:5d}  {t[i]:8.4f}  {tau_col[i]:10.2f}  {qpos_col[i]:12.2f}  {tgt_col[i]:13.2f}  {qpos_col[i]-tgt_col[i]:8.2f}{flag}")


def pinocchio_step_analysis(d, step):
    sys.path.insert(0, "/home/humanoid/ws_ctrl/src/h12_ros2_controller")
    import pinocchio as pin

    model, _, _ = pin.buildModelsFromUrdf(
        URDF_PATH, package_dirs=os.path.dirname(URDF_PATH),
        root_joint=pin.JointModelFreeFlyer()
    )
    data = model.createData()

    q27  = d["qpos"][step]
    t27  = d["tau"][step]
    fq   = np.zeros(model.nq); fq[3:7] = [0,0,0,1]; fq[7:] = q27
    tg   = pin.rnea(model, data, fq, np.zeros(model.nv), np.zeros(model.nv))[6:]

    pin.forwardKinematics(model, data, fq)
    pin.updateFramePlacements(model, data)

    def wrench_contrib(frame_name):
        fid = model.getFrameId(frame_name)
        J   = pin.computeFrameJacobian(model, data, fq, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, 6:]
        JpTinv = np.linalg.pinv(J.T)
        res = t27 - tg
        _, S, _ = np.linalg.svd(J)
        total = JpTinv @ res
        per_joint = {BODY_JOINTS[j]: JpTinv[:, j] * res[j] for j in range(27)}
        return total, per_joint, res, S

    print(f"\n{'─'*60}")
    print(f"  PINOCCHIO ANALYSIS  step={step}  t={d['time'][step]:.4f}s")

    for side, frame in [("LEFT", "left_wrist_yaw_link"), ("RIGHT", "right_wrist_yaw_link")]:
        total, per_j, res, S = wrench_contrib(frame)
        F_mag = np.linalg.norm(total[:3])
        print(f"\n  {side} WRIST WRENCH  [{frame}]")
        print(f"    Fx={total[0]:.2f}  Fy={total[1]:.2f}  Fz={total[2]:.2f}  |F|={F_mag:.2f} N")
        print(f"    Condition number: {S[0]/S[-1]:.2f}")
        print(f"    Top contributions to |Fz|:")
        contribs = [(name, c[2]) for name, c in per_j.items()]
        contribs.sort(key=lambda x: abs(x[1]), reverse=True)
        for name, fz in contribs[:5]:
            j = BODY_JOINTS.index(name)
            print(f"      {name:16s}: residual={res[j]:8.2f} Nm  Fz_contrib={fz:10.2f} N")


def main():
    parser = argparse.ArgumentParser(description="Explore a real-robot episode")
    parser.add_argument("--data", default="data/real/re3_real_encode_tr1")
    parser.add_argument("--summary", action="store_true", default=True,
                        help="Print episode summary (default: on)")
    parser.add_argument("--step", type=int, default=None,
                        help="Show detailed state at this timestep")
    parser.add_argument("--joints", nargs="+", default=None,
                        help="Show timeseries for specific joints (partial name match)")
    parser.add_argument("--pin", action="store_true",
                        help="Run Pinocchio wrench analysis at --step (requires pinocchio env)")
    args = parser.parse_args()

    d = load(args.data)

    if args.summary:
        summary(d)

    if args.step is not None:
        show_step(d, args.step)
        if args.pin:
            pinocchio_step_analysis(d, args.step)

    if args.joints:
        show_joint(d, args.joints)


if __name__ == "__main__":
    main()
