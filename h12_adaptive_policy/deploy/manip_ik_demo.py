"""
Single-arm pick-place with CLOSED-LOOP world-frame EE tracking + FAME.

The whole point: the controller tracks a target FIXED IN THE WORLD (a spot over the table), not a
pre-baked joint trajectory. Reads single_arm_manip.yaml (EE Cartesian waypoints, defined in the
base frame); these are mapped through the robot's NOMINAL base pose into a fixed world target path.
Every control tick the controller (Pink velocity IK on a reduced arm-only model) expresses that
world target in the CURRENT base frame (measured pelvis pose) and re-solves -> the arm actively
COMPENSATES base motion to keep the hand on the world target. The legs are driven by the FAME
policy; the passive arm hangs at arm_down; the payload hangs on the manip hand.

We measure, from privileged sim state:
  e^W_ee   = hand vs the fixed world target            -> world tracking error (what OptiTrack sees)
  e^W_base = base motion propagated to the hand        -> the disturbance the IK must reject
             (where the hand would land if the arm held its nominal base-frame pose)
With FAME the base stays put -> small disturbance -> the controller easily holds the world target.
Without FAME the base drifts (or falls) -> larger/faster disturbance -> the arm runs out of
reach/bandwidth to compensate and the world error grows. The base-frame view then shows the arm
deviating from its nominal pose to absorb the base motion.

Usage (from repo root):
  python h12_adaptive_policy/deploy/manip_ik_demo.py --task right_hand_manip \
      --video simulation_exp/videos/manip_right
  python h12_adaptive_policy/deploy/manip_ik_demo.py --task left_hand_manip --payload_kg 1.5
"""

import sys
import os
import argparse
import yaml
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
for p in (_SCRIPT_DIR, os.path.join(_REPO_ROOT, "h12_adaptive_policy"),
          os.path.join(_REPO_ROOT, "submodules", "h12_ros2_controller")):
    if p not in sys.path:
        sys.path.insert(0, p)

import mujoco
import torch
import pinocchio as pin
import pink
import qpsolvers
from mujoco_deploy_h12_rma import (
    load_config, load_safety_q_clip, pd_control, compute_observation,
    build_et_mujoco, get_gravity_orientation, RMA_LATENT_DIM,
)
from RMA.rma_modules.env_factor_encoder import EnvFactorEncoder, EnvFactorEncoderCfg
from h12_ros2_controller.core.robot_model import RobotModel

HEIGHT_THRESHOLD = 0.55
TILT_DEG_THRESHOLD = 45.0
_ARM_JN = ("shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint", "elbow_joint",
           "wrist_roll_joint", "wrist_pitch_joint", "wrist_yaw_joint")
ARM_JOINTS = {s: [f"{s}_{j}" for j in _ARM_JN] for s in ("left", "right")}
# slices into the 15-dof upper vector [torso, L_arm(7), R_arm(7)]
ARM_SLICE = {"left": slice(1, 8), "right": slice(8, 15)}


def quat2R(q):  # MuJoCo quat [w,x,y,z] -> rotation matrix
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def solve_arm_waypoints(rm, side, arm_down, torso, waypoints, orientation_cost=0.0):
    """Offline IK: arm-only (reduced model) joint targets for each EE waypoint. Returns (Nx7, resid_m)."""
    m = rm.model_body
    qn = pin.neutral(m)
    for s in ("left", "right"):
        for jname, val in zip(ARM_JOINTS[s], arm_down):
            qn[m.joints[m.getJointId(jname)].idx_q] = val
    qn[m.joints[m.getJointId("torso_joint")].idx_q] = torso
    lock = [m.getJointId(n) for n in m.names[1:] if n not in ARM_JOINTS[side]]
    rmod = pin.buildReducedModel(m, lock, qn)
    rdat = rmod.createData()
    cfg = pink.Configuration(rmod, rdat, np.asarray(arm_down, dtype=float))
    ft = pink.tasks.FrameTask(f"{side}_wrist_yaw_link", position_cost=50.0,
                              orientation_cost=orientation_cost, lm_damping=1.0)
    pt = pink.tasks.PostureTask(cost=1e-3)
    solver = next(s for s in ("daqp", "quadprog", "osqp", "proxqp") if s in qpsolvers.available_solvers)
    sols, resid = [], []
    for wp in waypoints:
        rpy = np.deg2rad(wp.get("rpy", [0, 0, 0]))
        R = pin.rpy.rpyToMatrix(*rpy) if orientation_cost > 0 else np.eye(3)
        ft.set_target(pin.SE3(R, np.asarray(wp["xyz"], float)))
        pt.set_target(cfg.q)
        for _ in range(800):
            cfg.integrate_inplace(pink.solve_ik(cfg, [ft, pt], dt=0.05, solver=solver), 0.05)
            if np.linalg.norm(ft.compute_error(cfg)[:3]) < 1e-3:
                break
        sols.append(cfg.q.copy())
        resid.append(float(np.linalg.norm(ft.compute_error(cfg)[:3])))
    return np.array(sols), np.array(resid)


def build_schedule(arm_down, sols, settle_s, seg_s, hold_s, dt):
    """Keyframe schedule (time, 7-joint) : settle at arm_down, then move through each waypoint."""
    keys = [(0.0, np.asarray(arm_down, float)), (settle_s, np.asarray(arm_down, float))]
    t = settle_s
    for q in sols:
        t += seg_s; keys.append((t, q))
        t += hold_s; keys.append((t, q))
    total = t + 0.5
    times = np.array([k[0] for k in keys])
    qs = np.stack([k[1] for k in keys])

    def arm_at(tt):
        if tt <= times[0]:
            return qs[0]
        if tt >= times[-1]:
            return qs[-1]
        i = int(np.searchsorted(times, tt) - 1)
        span = times[i + 1] - times[i]
        u = smoothstep((tt - times[i]) / span) if span > 1e-9 else 1.0
        return qs[i] + u * (qs[i + 1] - qs[i])

    return arm_at, total


class ArmIK:
    """Closed-loop velocity IK for the manipulating arm on a reduced (arm-only) model.
    step(base_target_xyz, dt) drives the wrist frame toward a target given in the CURRENT BASE frame
    and returns the 7 arm joint angles — re-solved every control tick so the arm actively compensates
    base motion to keep the hand on its world target (this is the controller half of the system)."""

    def __init__(self, rm, side, arm_down, torso):
        mb = rm.model_body
        qn = pin.neutral(mb)
        for s in ("left", "right"):
            for jn, v in zip(ARM_JOINTS[s], arm_down):
                qn[mb.joints[mb.getJointId(jn)].idx_q] = v
        qn[mb.joints[mb.getJointId("torso_joint")].idx_q] = torso
        lock = [mb.getJointId(n) for n in mb.names[1:] if n not in ARM_JOINTS[side]]
        self.model = pin.buildReducedModel(mb, lock, qn)
        self.data = self.model.createData()
        self.frame = f"{side}_wrist_yaw_link"
        self.cfg = pink.Configuration(self.model, self.data, np.asarray(arm_down, float))
        self.ft = pink.tasks.FrameTask(self.frame, position_cost=50.0, orientation_cost=0.0, lm_damping=1.0)
        self.pt = pink.tasks.PostureTask(cost=1e-3)
        self.solver = next(s for s in ("daqp", "quadprog", "osqp", "proxqp") if s in qpsolvers.available_solvers)

    def step(self, base_target, dt):
        self.ft.set_target(pin.SE3(np.eye(3), np.asarray(base_target, float)))
        self.pt.set_target(self.cfg.q)
        vel = pink.solve_ik(self.cfg, [self.ft, self.pt], dt=dt, solver=self.solver)
        self.cfg.integrate_inplace(vel, dt)
        return self.cfg.q.copy()


def build_xyz_schedule(waypoints, settle_s, seg_s, hold_s):
    """Base-frame EE target path vs time: hold at home through settle, then move home->...->place.
    Mapped through the nominal base pose this becomes the FIXED world target the controller tracks."""
    pts = [np.asarray(w["xyz"], float) for w in waypoints]
    keys = [(0.0, pts[0]), (settle_s, pts[0])]
    t = settle_s
    for p in pts:
        t += seg_s; keys.append((t, p))
        t += hold_s; keys.append((t, p))
    total = t + 0.5
    times = np.array([k[0] for k in keys]); P = np.stack([k[1] for k in keys])

    def xyz_at(tt):
        if tt <= times[0]:
            return P[0]
        if tt >= times[-1]:
            return P[-1]
        i = int(np.searchsorted(times, tt) - 1)
        span = times[i + 1] - times[i]
        u = smoothstep((tt - times[i]) / span) if span > 1e-9 else 1.0
        return P[i] + u * (P[i + 1] - P[i])

    return xyz_at, total


def build_payload_profile(settle_s):
    """On-the-fly payload (kg) vs time: SUDDEN step changes within the trained 0-3kg (<=30N) range,
    applied while the arm is moving. This is what exercises the encoder's online adaptation.
    Returns (payload_at, step_times)."""
    r = 0.15  # step duration (s): short = sudden change
    keys = [(0.0, 0.0), (settle_s, 0.0),
            (settle_s + 0.3, 1.5),                                 # initial grasp ~15N
            (settle_s + 2.3, 1.5), (settle_s + 2.3 + r, 3.0),      # +1.5kg suddenly -> ~30N (trained edge)
            (settle_s + 4.5, 3.0), (settle_s + 4.5 + r, 0.8),      # drop to ~8N (object removed)
            (settle_s + 6.5, 0.8), (settle_s + 6.5 + r, 2.5)]      # +1.7kg suddenly -> ~25N
    steps = [settle_s + 0.3, settle_s + 2.3, settle_s + 4.5, settle_s + 6.5]
    times = np.array([k[0] for k in keys]); V = np.array([k[1] for k in keys])

    def payload_at(t):
        if t <= times[0]:
            return V[0]
        if t >= times[-1]:
            return V[-1]
        i = int(np.searchsorted(times, t) - 1)
        span = times[i + 1] - times[i]
        u = smoothstep((t - times[i]) / span) if span > 1e-9 else 1.0
        return V[i] + u * (V[i + 1] - V[i])

    return payload_at, steps


def overlay(frame, lines):
    import cv2
    f = np.ascontiguousarray(frame); y = 24
    for txt, col in lines:
        cv2.putText(f, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA); y += 26
    return f


def _connector(scene, gtype, width, a, b, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.zeros(3), np.zeros(3), np.eye(3).reshape(9), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, gtype, width, np.asarray(a, np.float64), np.asarray(b, np.float64))
    scene.ngeom += 1


def add_force_arrow(scene, origin, force, scale=0.011, rgba=(1.0, 0.2, 0.1, 1.0)):
    """Red arrow at `origin` along `force` (length scaled by magnitude) — the payload weight."""
    if float(np.linalg.norm(force)) < 1e-6:
        return
    to = np.asarray(origin, np.float64) + np.asarray(force, np.float64) * scale
    _connector(scene, mujoco.mjtGeom.mjGEOM_ARROW, 0.02, origin, to, rgba)


def add_marker(scene, pos, radius=0.045, rgba=(0.1, 0.95, 0.2, 1.0)):
    """Sphere at `pos` — the commanded EE target (so the gap to the hand = world EE error)."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0], np.float64),
                        np.asarray(pos, np.float64), np.eye(3).reshape(9), np.asarray(rgba, np.float32))
    scene.ngeom += 1


def add_error_line(scene, hand, target, rgba=(1.0, 0.95, 0.1, 1.0)):
    """Yellow rod from the hand to the commanded target = the world EE error vector."""
    if float(np.linalg.norm(np.asarray(hand) - np.asarray(target))) < 1e-4:
        return
    _connector(scene, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.007, hand, target, rgba)


def run_manip(config, m, ids, policy, encoder, bounds, *, side, rm, xyz_at, total_s, payload_kg,
              settle_s, load_ramp_s, torso, no_encode, payload_at=None, renderer=None, cam=None,
              render_stride=20, label="", live_view=False):
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    dt = config["simulation_dt"]; decim = config["control_decimation"]
    policy_joints = int(config.get("policy_num_joints", 27)); h12 = int(config.get("h12_ctrl_count", 27))
    leg_count = config["num_actions"]; upper_n = h12 - leg_count
    jids = m.actuator_trnid[:h12, 0].astype(np.int32)
    qadr = m.jnt_qposadr[jids].astype(np.int32); vadr = m.jnt_dofadr[jids].astype(np.int32)
    leg_qadr, leg_vadr = qadr[:leg_count], vadr[:leg_count]
    up_qadr, up_vadr = qadr[leg_count:h12], vadr[leg_count:h12]
    lwid, rwid, manip_eid = ids["left_wrist"], ids["right_wrist"], ids[f"{side}_ee"]

    # nominal pelvis pose (world) and nominal hand-in-pelvis -> world target reference
    p0 = d.qpos[:3].copy(); R0 = quat2R(d.qpos[3:7])
    action = np.zeros(leg_count, dtype=np.float32)
    target_dof_pos = config["default_angles"].copy()
    cmd = config["cmd_init"].copy(); height_cmd = float(config["height_cmd"])
    arm_down = np.asarray(config.get("_arm_down"), dtype=np.float32)
    ik = ArmIK(rm, side, arm_down, torso)                       # closed-loop world-frame controller
    arm_q_hold = np.asarray(arm_down, dtype=np.float32).copy()  # latest IK solution (held between control ticks)

    qj = d.qpos[qadr]; dqj = d.qvel[vadr]
    sobs, _ = compute_observation(d, config, action, cmd, height_cmd, policy_joints, qj=qj, dqj=dqj)
    import collections
    obs_hist = collections.deque([sobs.copy()] * config["obs_history_len"], maxlen=config["obs_history_len"])
    z_hist = np.zeros((3, RMA_LATENT_DIM), dtype=np.float32)
    g = np.array([0, 0, -9.81]); max_tau = 200.0
    n_steps = int(total_s / dt); ss_start = max(0, n_steps - int(round(1.0 / dt)))
    sc = bounds["safety_clip"]; lql, lqh = bounds["leg_q_low"], bounds["leg_q_high"]
    uql, uqh = bounds["upper_q_low"], bounds["upper_q_high"]

    ssq_base = ssq_ee = ssq_eeb = 0.0; max_ee = max_tilt = 0.0; ss_ee = 0.0; nacc = ssn = 0; fell = False
    e_ee = e_ee_b = 0.0
    d0q = d.qpos.copy(); d_ref = mujoco.MjData(m)   # reference for commanded-EE forward kinematics
    traj = {"t": [], "world": [], "cmd_world": [], "bpos": [], "bquat": [], "load": [], "z": []}
    last_cmd_world = None        # green marker: fixed world target the controller holds
    last_dist_world = None       # magenta marker: where base motion alone would put the hand (disturbance)
    frames = [] if renderer is not None else None
    import time as _time
    viewer = None
    if live_view:
        import mujoco.viewer as _mjv
        viewer = _mjv.launch_passive(m, d)

    for step in range(n_steps):
        _t0 = _time.time()
        t = step * dt
        # --- payload on the manip hand: constant (ramped at grasp) OR a time-varying on-the-fly profile ---
        if payload_at is not None:
            kg_now = float(payload_at(t)); lr = 1.0
        else:
            lr = smoothstep((t - settle_s) / max(load_ramp_s, 1e-6)); kg_now = lr * payload_kg
        F = kg_now * g
        d.xfrc_applied[:] = 0
        d.xfrc_applied[manip_eid, :3] = F
        ef = F.astype(np.float32)
        ef_left = ef if side == "left" else np.zeros(3, np.float32)
        ef_right = ef if side == "right" else np.zeros(3, np.float32)

        # --- manip arm: closed-loop IK tracks a FIXED WORLD target, compensating the base motion ---
        # world target = nominal-base pose applied to the base-frame waypoint path (fixed in the room).
        # Each control tick: express it in the CURRENT base frame and re-solve IK -> the arm moves to
        # keep the hand on the world target even as the base drifts (no-FAME) or stays put (FAME).
        if step % decim == 0:
            pelvis_c = d.qpos[:3].copy(); Rp_c = quat2R(d.qpos[3:7])
            world_tgt = R0 @ xyz_at(t) + p0
            base_tgt = Rp_c.T @ (world_tgt - pelvis_c)
            arm_q_hold = ik.step(base_tgt, dt * decim).astype(np.float32)
        arm_cmd = np.zeros(upper_n, dtype=np.float32)
        arm_cmd[0] = torso
        arm_cmd[ARM_SLICE[side]] = arm_q_hold
        arm_cmd[ARM_SLICE["left" if side == "right" else "right"]] = arm_down

        if sc:
            target_dof_pos = np.clip(target_dof_pos, lql, lqh)
        leg_tau = pd_control(target_dof_pos, d.qpos[leg_qadr], config["kps"],
                             np.zeros_like(config["kps"]), d.qvel[leg_vadr], config["kds"])
        d.ctrl[:leg_count] = np.clip(np.nan_to_num(leg_tau), -max_tau, max_tau)
        if upper_n > 0:
            kpa = config.get("kps_arms", np.ones(upper_n) * 500.0); kda = config.get("kds_arms", np.ones(upper_n) * 5.0)
            at = np.clip(arm_cmd, uql, uqh) if sc else arm_cmd
            arm_tau = pd_control(at, d.qpos[up_qadr], kpa, np.zeros(upper_n), d.qvel[up_vadr], kda)
            d.ctrl[leg_count:h12] = np.clip(np.nan_to_num(arm_tau), -max_tau, max_tau)
        if d.ctrl.shape[0] > h12:
            gr = m.actuator_ctrlrange[h12:, :]; d.ctrl[h12:] = 0.5 * (gr[:, 0] + gr[:, 1])

        mujoco.mj_step(m, d)

        # --- base drift (every step); EE decomposition recorded at control rate below ---
        pelvis = d.qpos[:3]; Rp = quat2R(d.qpos[3:7]); ee_w = d.xpos[manip_eid]
        e_base = float(np.linalg.norm(pelvis[:2] - p0[:2]))
        tilt = float(np.degrees(np.arccos(np.clip(-get_gravity_orientation(d.qpos[3:7])[2], -1, 1))))
        ssq_base += e_base ** 2; max_tilt = max(max_tilt, tilt); nacc += 1
        if d.qpos[2] < HEIGHT_THRESHOLD or tilt > TILT_DEG_THRESHOLD:
            fell = True

        if renderer is not None and step % render_stride == 0:
            renderer.update_scene(d, camera=cam)
            if last_cmd_world is not None:
                # magenta rod = base disturbance the IK rejects (where the hand would be w/o compensation)
                if last_dist_world is not None:
                    add_error_line(renderer.scene, last_dist_world, last_cmd_world, rgba=(1.0, 0.25, 1.0, 0.9))
                    add_marker(renderer.scene, last_dist_world, radius=0.038, rgba=(1.0, 0.2, 1.0, 0.95))
                add_marker(renderer.scene, last_cmd_world)            # green: fixed world target (hand holds it)
            add_force_arrow(renderer.scene, ee_w, F)                  # red: payload weight
            l2 = (f"t={t:4.1f}s  load={kg_now:.1f}kg  pelvis drift={e_base*100:4.1f}cm  "
                  f"world err={e_ee*100:4.1f}cm  disturbance={e_ee_b*100:4.1f}cm")
            lines = [(label, (255, 255, 255)), (l2, (255, 255, 255)),
                     ("green=world target (held)   magenta=base disturbance IK rejects   red=load", (180, 230, 160))]
            if fell:
                lines.append(("FELL", (255, 60, 60)))
            frames.append(overlay(renderer.render(), lines))

        if step % decim == 0:
            # commanded EE = the FIXED WORLD target the controller is tracking (nominal base x base-path).
            #   e^W_ee   = actual hand vs world target               -> world tracking error (headline)
            #   e^W_base = base motion propagated to the hand        -> the disturbance the IK rejects /
            #              (= where the hand would be if the arm held its nominal base-frame pose)
            cmd_world = R0 @ xyz_at(t) + p0
            dist_world = Rp @ xyz_at(t) + pelvis
            last_cmd_world = cmd_world; last_dist_world = dist_world
            e_ee = float(np.linalg.norm(ee_w - cmd_world))
            e_ee_b = float(np.linalg.norm(dist_world - cmd_world))
            ssq_ee += e_ee ** 2; ssq_eeb += e_ee_b ** 2; max_ee = max(max_ee, e_ee)
            if step >= ss_start:
                ss_ee += e_ee; ssn += 1
            traj["t"].append(t); traj["world"].append(ee_w.copy()); traj["cmd_world"].append(cmd_world)
            traj["bpos"].append(pelvis.copy()); traj["bquat"].append(d.qpos[3:7].copy())
            traj["load"].append(kg_now)
            qj = d.qpos[qadr]; dqj = d.qvel[vadr]
            sobs, _ = compute_observation(d, config, action, cmd, height_cmd, policy_joints, qj=qj, dqj=dqj)
            obs_hist.append(sobs)
            e_t = build_et_mujoco(d.qpos, np.zeros(3) if no_encode else ef_left,
                                  np.zeros(3) if no_encode else ef_right, leg_count, policy_joints, qj)
            with torch.no_grad():
                z = encoder(torch.from_numpy(e_t).unsqueeze(0).float()).numpy().squeeze()
            z_hist[1:] = z_hist[:-1]; z_hist[0] = z
            traj["z"].append(float(np.linalg.norm(z)))   # encoder latent magnitude (FAME adapts; blind ~constant)
            actor_obs = np.concatenate([np.concatenate(list(obs_hist)), np.flip(z_hist, 0).flatten()]).astype(np.float32)
            action = policy(torch.from_numpy(actor_obs).unsqueeze(0)).detach().numpy().squeeze()
            target_dof_pos = action * config["action_scale"] + config["default_angles"]

        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            _rem = dt - (_time.time() - _t0)
            if _rem > 0:
                _time.sleep(_rem)

    if viewer is not None:
        viewer.close()
    nee = max(1, len(traj["t"]))
    for k in list(traj):
        traj[k] = np.array(traj[k])
    return {
        "ee_rmse": float(np.sqrt(ssq_ee / nee)), "ee_max": max_ee, "ee_ss": ss_ee / max(1, ssn),
        "ee_b_rmse": float(np.sqrt(ssq_eeb / nee)),
        "base_rmse": float(np.sqrt(ssq_base / nacc)), "tilt_max": max_tilt, "fell": int(fell),
        "traj": traj,
    }, frames


def _base_frame(traj):
    """actual & commanded EE expressed in the base frame (current base for actual, nominal for commanded)."""
    p0 = traj["bpos"][0]; R0 = quat2R(traj["bquat"][0])
    act_b = np.array([quat2R(q).T @ (w - p) for w, p, q in zip(traj["world"], traj["bpos"], traj["bquat"])])
    cmd_b = np.array([R0.T @ (c - p0) for c in traj["cmd_world"]])
    return act_b, cmd_b


def plot_traj(trajs, payload, task, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fame, nof = trajs["FAME"], trajs["no-FAME"]
    t = fame["t"]; cmd_w = fame["cmd_world"]
    fa_b, cmd_b = _base_frame(fame); nf_b, _ = _base_frame(nof)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    names = ["x (forward)", "y (left)", "z (up)"]
    for j in range(3):
        a0, a1 = axes[0, j], axes[1, j]
        a0.plot(t, cmd_w[:, j] * 100, "k-", lw=2, label="commanded")
        a0.plot(t, fame["world"][:, j] * 100, color="#2c7fb8", lw=1.6, label="FAME actual")
        a0.plot(t, nof["world"][:, j] * 100, color="#d95f0e", ls="--", lw=1.6, label="no-FAME actual")
        a0.set_title(f"WORLD  {names[j]}"); a0.grid(alpha=.3)
        a1.plot(t, cmd_b[:, j] * 100, "k-", lw=2)
        a1.plot(t, fa_b[:, j] * 100, color="#2c7fb8", lw=1.6)
        a1.plot(t, nf_b[:, j] * 100, color="#d95f0e", ls="--", lw=1.6)
        a1.set_title(f"BASE  {names[j]}"); a1.grid(alpha=.3); a1.set_xlabel("t (s)")
    axes[0, 0].set_ylabel("position (cm)"); axes[1, 0].set_ylabel("position (cm)")
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle(f"{task}, {payload} kg — fixed WORLD target vs actual hand (closed-loop tracking)\n"
                 "WORLD frame: the hand holds the fixed world target — the controller compensates base motion (e^W_ee small).   "
                 "BASE frame: the arm DEVIATES from its nominal command to absorb base drift — more for no-FAME",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(out, dpi=150); print(f"saved figure -> {out}")


def plot_summary(results, payloads, task, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    styles = {"FAME": ("#2c7fb8", "o", "-"), "no-FAME": ("#d95f0e", "s", "--")}
    for ax, key, title in [(axes[0], "ee_rmse", r"world tracking error  $e^W_{ee}$  (after control)"),
                           (axes[1], "ee_b_rmse", r"base disturbance at hand  $e^W_{base}$  (IK rejects this)"),
                           (axes[2], "base_rmse", "pelvis drift (horizontal)")]:
        for lab, (c, mk, ls) in styles.items():
            y = [r[key] * 100 for r in results[lab]]
            ax.plot(payloads, y, ls, color=c, marker=mk, label=lab)
            xf = [p for p, r in zip(payloads, results[lab]) if r["fell"]]
            if xf:
                ax.scatter(xf, [max(y)] * len(xf), marker="x", color="red", zorder=5)
        ax.set_xlabel("payload (kg)"); ax.set_ylabel("cm"); ax.set_title(title); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.suptitle(f"{task}: closed-loop world-frame tracking vs payload — FAME shrinks the disturbance the IK must reject",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92]); fig.savefig(out, dpi=150); print(f"saved figure -> {out}")


def plot_adapt(trajs, steps, task, out):
    """Time series under an ON-THE-FLY changing payload: load, world EE error, pelvis drift —
    FAME (force encoder, adapts) vs no-FAME (blind). Vertical lines = sudden load changes."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fame, nof = trajs["FAME"], trajs["no-FAME"]
    t = fame["t"]
    werr = lambda tr: np.linalg.norm(tr["world"] - tr["cmd_world"], axis=1) * 100
    drift = lambda tr: np.linalg.norm(tr["bpos"][:, :2] - tr["bpos"][0, :2], axis=1) * 100
    fig, axes = plt.subplots(4, 1, figsize=(12, 10.5), sharex=True)
    axes[0].plot(t, fame["load"], "k-", lw=2); axes[0].set_ylabel("payload (kg)")
    axes[0].axhline(3.06, color="gray", ls=":", lw=1); axes[0].text(t[2], 3.1, "30 N trained limit", fontsize=8, color="gray")
    axes[0].set_title(f"{task}: payload changed ON THE FLY while moving — FAME adapts (force encoder) vs no-FAME (blind)")
    axes[1].plot(t, fame["z"], color="#2c7fb8", lw=1.7, label="FAME (force encoder)")
    axes[1].plot(t, nof["z"], color="#d95f0e", ls="--", lw=1.7, label="no-FAME (blind)")
    axes[1].set_ylabel(r"encoder latent $\|z\|$"); axes[1].legend(fontsize=9, loc="upper left")
    for ax, fn, ylab in [(axes[2], werr, "world EE error (cm)"), (axes[3], drift, "pelvis drift (cm)")]:
        ax.plot(t, fn(fame), color="#2c7fb8", lw=1.7, label="FAME (adapts)")
        ax.plot(t, fn(nof), color="#d95f0e", ls="--", lw=1.7, label="no-FAME (blind)")
        ax.set_ylabel(ylab); ax.legend(fontsize=9, loc="upper left")
    axes[3].set_xlabel("t (s)")
    for ax in axes:
        for s in steps:
            ax.axvline(s, color="purple", ls=":", lw=1.2, alpha=0.6)
        ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); print(f"saved figure -> {out}")


def main():
    p = argparse.ArgumentParser(description="Single-arm IK pick-place + FAME: world-EE decomposition demo")
    p.add_argument("--config", default=os.path.join(_SCRIPT_DIR, "h1_2_rma_arm_magpie_fame.yaml"))
    p.add_argument("--manip_yaml", default=os.path.join(_SCRIPT_DIR, "single_arm_manip.yaml"))
    p.add_argument("--task", default="right_hand_manip", choices=("right_hand_manip", "left_hand_manip"))
    p.add_argument("--payload_kg", type=float, default=None, help="Override the YAML payload")
    p.add_argument("--view", action="store_true", help="Open a live MuJoCo viewer (one condition, real-time)")
    p.add_argument("--view_cond", default="FAME", choices=("FAME", "no-FAME"), help="Condition to show in --view")
    p.add_argument("--sweep", action="store_true", help="Sweep payload 1-6kg; save commanded-vs-actual trajectory + summary figures")
    p.add_argument("--sweep_video", action="store_true", help="Render ONE side-by-side video stepping through 1-6kg")
    p.add_argument("--adapt", action="store_true", help="On-the-fly payload changes (steps in 0-3kg) while moving: FAME vs no-FAME")
    p.add_argument("--plot_payload", type=int, default=4, help="Payload (kg) for the trajectory plot")
    p.add_argument("--video", default=None, help="path prefix for FAME/no-FAME/side-by-side mp4s")
    p.add_argument("--vid_w", type=int, default=640); p.add_argument("--vid_h", type=int, default=480)
    p.add_argument("--fps", type=int, default=25); p.add_argument("--render_stride", type=int, default=20)
    args = p.parse_args()

    mc = yaml.safe_load(open(args.manip_yaml))
    task = mc[args.task]; side = task["manip"]
    arm_down = np.asarray(mc["arm_down"], dtype=np.float32); torso = float(mc.get("torso", -0.35))
    payload = args.payload_kg if args.payload_kg is not None else float(task.get("payload_kg", 1.0))
    settle_s, seg_s, hold_s, ramp_s = (float(mc.get(k, v)) for k, v in
                                       [("load_ramp_s", 0.8), ("seg_time_s", 1.5), ("hold_s", 0.8), ("load_ramp_s", 0.8)])
    settle_s = 1.0
    oc = float(mc.get("orientation_cost", 0.0)) if mc.get("track_orientation") else 0.0

    # --- config / model / policy / encoder / safety ---
    config = load_config(args.config); cdir = os.path.dirname(os.path.abspath(args.config))
    for k in ("policy_path", "xml_path", "encoder_path"):
        if config.get(k) and not os.path.isabs(config[k]):
            config[k] = os.path.normpath(os.path.join(cdir, config[k]))
    config["_arm_down"] = arm_down
    m = mujoco.MjModel.from_xml_path(config["xml_path"]); m.opt.timestep = config["simulation_dt"]
    ids = {"left_wrist": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_roll_link"),
           "right_wrist": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link"),
           "left_ee": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link"),
           "right_ee": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")}
    policy = torch.jit.load(config["policy_path"]); policy.eval()
    encoder = EnvFactorEncoder(EnvFactorEncoderCfg())
    encoder.load_state_dict(torch.load(config["encoder_path"], map_location="cpu", weights_only=True)); encoder.eval()
    sc = bool(config.get("safety_clip", True)); bounds = dict(safety_clip=sc, leg_q_low=None, leg_q_high=None, upper_q_low=None, upper_q_high=None)
    if sc:
        lc = config["num_actions"]; hc = int(config.get("h12_ctrl_count", 27))
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in m.actuator_trnid[:hc, 0].astype(int)]
        qlo, qhi = load_safety_q_clip(names, config.get("safety_config"))
        bounds.update(leg_q_low=qlo[:lc], leg_q_high=qhi[:lc], upper_q_low=qlo[lc:hc], upper_q_high=qhi[lc:hc])

    # --- IK: arm joint waypoints for the EE targets ---
    rm = RobotModel(os.path.join(_REPO_ROOT, "submodules/h12_ros2_controller/assets/h1_2/h1_2_handless.urdf"), handless=True)
    sols, resid = solve_arm_waypoints(rm, side, arm_down, torso, task["waypoints"], orientation_cost=oc)
    print(f"[IK] task={args.task} side={side} payload={payload}kg")
    for wp, r in zip(task["waypoints"], resid):
        flag = "" if r < 0.01 else "  <-- UNREACHABLE (residual high)"
        print(f"   {wp['name']:6s} xyzsubscribe to rt/lowcmd and publish to rt/lowstate={wp['xyz']}  IK residual={r*1000:5.1f}mm{flag}")
    xyz_at, total = build_xyz_schedule(task["waypoints"], settle_s, seg_s, hold_s)

    if args.view:
        print(f"\n[live view] {args.task} — {args.view_cond}  (close the window to stop)")
        run_manip(config, m, ids, policy, encoder, bounds, side=side, rm=rm, xyz_at=xyz_at, total_s=total,
                  payload_kg=payload, settle_s=settle_s, load_ramp_s=ramp_s, torso=torso,
                  no_encode=(args.view_cond == "no-FAME"), live_view=True,
                  label=f"{args.view_cond} ({args.task}, {payload}kg)")
        return

    if args.sweep:
        payloads = list(range(1, 4))   # trained range only: 30 N ~= 3 kg (4-6 kg is out-of-distribution)
        results = {"FAME": [], "no-FAME": []}; trajs = {}
        print(f"\n[sweep] {args.task}: payloads {payloads} kg")
        for kg in payloads:
            for lab, no_enc in [("FAME", False), ("no-FAME", True)]:
                mt, _ = run_manip(config, m, ids, policy, encoder, bounds, side=side, rm=rm, xyz_at=xyz_at, total_s=total,
                                  payload_kg=float(kg), settle_s=settle_s, load_ramp_s=ramp_s, torso=torso, no_encode=no_enc)
                results[lab].append(mt)
                if kg == args.plot_payload:
                    trajs[lab] = mt["traj"]
                print(f"  {kg}kg {lab:8s} e^W_ee={mt['ee_rmse']*100:5.1f}cm  e^W_base={mt['ee_b_rmse']*100:4.1f}cm  "
                      f"base={mt['base_rmse']*100:4.1f}cm  fell={mt['fell']}")
        figdir = os.path.join(_REPO_ROOT, "simulation_exp/figures"); os.makedirs(figdir, exist_ok=True)
        if "FAME" in trajs and "no-FAME" in trajs:
            plot_traj(trajs, args.plot_payload, args.task, os.path.join(figdir, f"manip_{side}_traj_{args.plot_payload}kg.png"))
        plot_summary(results, payloads, args.task, os.path.join(figdir, f"manip_{side}_summary.png"))
        return

    if args.sweep_video:
        import imageio
        payloads = list(range(1, 4))   # trained range only: 30 N ~= 3 kg (4-6 kg is out-of-distribution)
        renderer = mujoco.Renderer(m, height=args.vid_h, width=args.vid_w)
        cam = mujoco.MjvCamera(); cam.lookat[:] = [0.1, 0, 0.9]; cam.distance = 3.2; cam.azimuth = 150; cam.elevation = -12
        all_sbs = []
        print(f"\n[sweep-video] {args.task}: stepping payloads {payloads} kg (no-FAME | FAME)")
        for kg in payloads:
            clips = {}; met = {}
            for lab, no_enc in [("FAME", False), ("no-FAME", True)]:
                mt, frames = run_manip(config, m, ids, policy, encoder, bounds, side=side, rm=rm, xyz_at=xyz_at,
                                       total_s=total, payload_kg=float(kg), settle_s=settle_s, load_ramp_s=ramp_s,
                                       torso=torso, no_encode=no_enc, renderer=renderer, cam=cam,
                                       render_stride=args.render_stride, label=f"{lab}  ({args.task}, {kg}kg)")
                clips[lab] = frames; met[lab] = mt
            print(f"  {kg}kg  e^W_ee FAME={met['FAME']['ee_rmse']*100:4.1f}cm no-FAME={met['no-FAME']['ee_rmse']*100:4.1f}cm  "
                  f"| pelvis drift FAME={met['FAME']['base_rmse']*100:4.1f}cm no-FAME={met['no-FAME']['base_rmse']*100:4.1f}cm")
            n = min(len(clips["FAME"]), len(clips["no-FAME"]))
            div = np.full((clips["FAME"][0].shape[0], 4, 3), 255, np.uint8)
            all_sbs.extend(np.concatenate([clips["no-FAME"][i], div, clips["FAME"][i]], axis=1) for i in range(n))
        renderer.close()
        out = args.video or os.path.join(_REPO_ROOT, f"simulation_exp/videos/manip_{side}_world_sweep")
        sp = f"{out}_sidebyside.mp4"; os.makedirs(os.path.dirname(sp) or ".", exist_ok=True)
        imageio.mimwrite(sp, all_sbs, fps=args.fps, codec="libx264", quality=8)
        print(f"saved {sp}  (1-6kg sweep, no-FAME | FAME)")
        return

    if args.adapt:
        import imageio
        payload_at, steps = build_payload_profile(settle_s)
        renderer = mujoco.Renderer(m, height=args.vid_h, width=args.vid_w)
        cam = mujoco.MjvCamera(); cam.lookat[:] = [0.1, 0, 0.9]; cam.distance = 3.2; cam.azimuth = 150; cam.elevation = -12
        trajs = {}; clips = {}
        print(f"\n[adapt] {args.task}: ON-THE-FLY payload steps at t={[round(s, 1) for s in steps]}s (within 0-3kg / 30N)")
        for lab, no_enc in [("FAME", False), ("no-FAME", True)]:
            mt, frames = run_manip(config, m, ids, policy, encoder, bounds, side=side, rm=rm, xyz_at=xyz_at,
                                   total_s=total, payload_kg=0.0, payload_at=payload_at, settle_s=settle_s,
                                   load_ramp_s=ramp_s, torso=torso, no_encode=no_enc, renderer=renderer, cam=cam,
                                   render_stride=args.render_stride, label=f"{lab}  on-the-fly load  ({args.task})")
            trajs[lab] = mt["traj"]; clips[lab] = frames
            print(f"  {lab:8s} world EE: rmse={mt['ee_rmse']*100:4.1f}cm peak={mt['ee_max']*100:5.1f}cm | "
                  f"pelvis drift rmse={mt['base_rmse']*100:4.1f}cm tilt={mt['tilt_max']:4.1f} fell={mt['fell']}")
        renderer.close()
        figdir = os.path.join(_REPO_ROOT, "simulation_exp/figures"); os.makedirs(figdir, exist_ok=True)
        plot_adapt(trajs, steps, args.task, os.path.join(figdir, f"manip_{side}_adapt.png"))
        n = min(len(clips["FAME"]), len(clips["no-FAME"]))
        div = np.full((clips["FAME"][0].shape[0], 4, 3), 255, np.uint8)
        sbs = [np.concatenate([clips["no-FAME"][i], div, clips["FAME"][i]], axis=1) for i in range(n)]
        out = args.video or os.path.join(_REPO_ROOT, f"simulation_exp/videos/manip_{side}_adapt")
        sp = f"{out}_sidebyside.mp4"; os.makedirs(os.path.dirname(sp) or ".", exist_ok=True)
        imageio.mimwrite(sp, sbs, fps=args.fps, codec="libx264", quality=8)
        print(f"saved {sp}  (on-the-fly load, no-FAME | FAME)")
        return

    renderer = cam = None
    if args.video:
        renderer = mujoco.Renderer(m, height=args.vid_h, width=args.vid_w)
        cam = mujoco.MjvCamera(); cam.lookat[:] = [0.1, 0, 0.9]; cam.distance = 3.2; cam.azimuth = 150; cam.elevation = -12

    clips = {}
    print(f"\n================ {args.task}: world-frame EE decomposition (FAME vs no-FAME) ================")
    for lab, no_enc in [("FAME", False), ("no-FAME", True)]:
        mt, frames = run_manip(config, m, ids, policy, encoder, bounds, side=side, rm=rm, xyz_at=xyz_at, total_s=total,
                               payload_kg=payload, settle_s=settle_s, load_ramp_s=ramp_s, torso=torso,
                               no_encode=no_enc, renderer=renderer, cam=cam, render_stride=args.render_stride,
                               label=f"{lab}  ({args.task}, {payload}kg)")
        clips[lab] = frames
        print(f"  {lab:8s} e^W_ee (world track): rmse={mt['ee_rmse']*100:5.1f}cm max={mt['ee_max']*100:5.1f}cm "
              f"| e^W_base (disturb)={mt['ee_b_rmse']*100:5.1f}cm  pelvis drift={mt['base_rmse']*100:4.1f}cm "
              f"tilt={mt['tilt_max']:4.1f} fell={mt['fell']}")
    print("=================================================================================================")

    if args.video:
        import imageio
        for lab, key in [("no-FAME", "no_fame"), ("FAME", "fame")]:
            path = f"{args.video}_{key}.mp4"; os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            imageio.mimwrite(path, clips[lab], fps=args.fps, codec="libx264", quality=8); print(f"saved {path}")
        n = min(len(clips["FAME"]), len(clips["no-FAME"]))
        div = np.full((clips["FAME"][0].shape[0], 4, 3), 255, np.uint8)
        sbs = [np.concatenate([clips["no-FAME"][i], div, clips["FAME"][i]], axis=1) for i in range(n)]
        sp = f"{args.video}_sidebyside.mp4"; imageio.mimwrite(sp, sbs, fps=args.fps, codec="libx264", quality=8)
        renderer.close(); print(f"saved {sp}  (no-FAME | FAME)")


if __name__ == "__main__":
    main()
