"""
Single-arm pick-place demo: track a WORLD-fixed EE target while the FAME policy walks
the legs. Pink velocity-IK re-solves every tick to compensate base motion; the passive
arm hangs, the manip hand carries a payload. Reports e^W_ee (world tracking error) and
e^W_base (the base-motion disturbance the IK must reject) — FAME vs no-FAME.

Run modes:
  default        one FAME vs no-FAME run, optional --video mp4
  --view         live MuJoCo viewer (one condition, real-time)
  --sweep        payload 1-3 kg, save trajectory + summary figures
  --sweep_video  side-by-side mp4 across the payload sweep
  --adapt        on-the-fly payload steps; FAME adapts, no-FAME blind
"""
import sys
import os
import argparse
import collections
import time
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
from mujoco_deploy_h12_rma import (
    load_config, load_safety_q_clip, pd_control, compute_observation,
    build_et_mujoco, get_gravity_orientation, RMA_LATENT_DIM,
)
from RMA.rma_modules.env_factor_encoder import EnvFactorEncoder, EnvFactorEncoderCfg
from h12_ros2_controller.core.robot_model import RobotModel
from utils import (
    quat2R, smoothstep,
    ArmIK, solve_arm_waypoints, build_xyz_schedule, build_payload_profile, ARM_SLICE,
    make_manip_camera, overlay_text, add_force_arrow, add_marker, add_error_line,
    save_mp4, save_sidebyside_mp4, sidebyside_frames,
    plot_traj, plot_summary, plot_adapt,
)

HEIGHT_THRESHOLD = 0.55
TILT_DEG_THRESHOLD = 45.0


def run_manip(config, m, ids, policy, encoder, bounds, *, side, rm, xyz_at, total_s, payload_kg,
              settle_s, load_ramp_s, torso, no_encode, payload_at=None, renderer=None, cam=None,
              render_stride=20, label="", live_view=False, seed=None, init_jitter=0.01):
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    # Multi-seed support: jitter the initial actuated-joint positions by Gaussian
    # noise (std = init_jitter rad). Breaks determinism so multiple trials yield
    # different trajectories without changing the task semantics.
    if seed is not None:
        rng = np.random.default_rng(int(seed))
        h12_for_jitter = int(config.get("h12_ctrl_count", 27))
        jids_for_jitter = m.actuator_trnid[:h12_for_jitter, 0].astype(np.int32)
        qadr_for_jitter = m.jnt_qposadr[jids_for_jitter].astype(np.int32)
        d.qpos[qadr_for_jitter] += rng.normal(0.0, init_jitter, size=h12_for_jitter)
        mujoco.mj_forward(m, d)
    dt = config["simulation_dt"]; decim = config["control_decimation"]
    policy_joints = int(config.get("policy_num_joints", 27)); h12 = int(config.get("h12_ctrl_count", 27))
    leg_count = config["num_actions"]; upper_n = h12 - leg_count
    jids = m.actuator_trnid[:h12, 0].astype(np.int32)
    qadr = m.jnt_qposadr[jids].astype(np.int32); vadr = m.jnt_dofadr[jids].astype(np.int32)
    leg_qadr, leg_vadr = qadr[:leg_count], vadr[:leg_count]
    up_qadr, up_vadr = qadr[leg_count:h12], vadr[leg_count:h12]
    lwid, rwid, manip_eid = ids["left_wrist"], ids["right_wrist"], ids[f"{side}_ee"]

    # nominal pelvis pose (world) -> fixed world target reference
    p0 = d.qpos[:3].copy(); R0 = quat2R(d.qpos[3:7])
    action = np.zeros(leg_count, dtype=np.float32)
    target_dof_pos = config["default_angles"].copy()
    cmd = config["cmd_init"].copy(); height_cmd = float(config["height_cmd"])
    arm_down = np.asarray(config.get("_arm_down"), dtype=np.float32)
    ik = ArmIK(rm, side, arm_down, torso)                       # closed-loop world-frame controller
    arm_q_hold = np.asarray(arm_down, dtype=np.float32).copy()  # latest IK solution (held between control ticks)

    qj = d.qpos[qadr]; dqj = d.qvel[vadr]
    sobs, _ = compute_observation(d, config, action, cmd, height_cmd, policy_joints, qj=qj, dqj=dqj)
    obs_hist = collections.deque([sobs.copy()] * config["obs_history_len"], maxlen=config["obs_history_len"])
    z_hist = np.zeros((3, RMA_LATENT_DIM), dtype=np.float32)
    g = np.array([0, 0, -9.81]); max_tau = 200.0
    n_steps = int(total_s / dt); ss_start = max(0, n_steps - int(round(1.0 / dt)))
    sc = bounds["safety_clip"]; lql, lqh = bounds["leg_q_low"], bounds["leg_q_high"]
    uql, uqh = bounds["upper_q_low"], bounds["upper_q_high"]

    ssq_base = ssq_ee = ssq_eeb = 0.0; max_ee = max_tilt = 0.0; ss_ee = 0.0; nacc = ssn = 0; fell = False
    e_ee = e_ee_b = 0.0
    traj = {"t": [], "world": [], "cmd_world": [], "bpos": [], "bquat": [], "load": [], "z": []}
    last_cmd_world = None        # green marker: fixed world target the controller holds
    last_dist_world = None       # magenta marker: where base motion alone would put the hand
    frames = [] if renderer is not None else None
    viewer = None
    if live_view:
        import mujoco.viewer as _mjv
        viewer = _mjv.launch_passive(m, d)

    for step in range(n_steps):
        _t0 = time.time()
        t = step * dt
        # --- payload on the manip hand: constant (ramped at grasp) OR a time-varying profile ---
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

        # --- manip arm: closed-loop IK tracks a FIXED WORLD target, compensating base motion ---
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
            kpa = config.get("kps_arms", np.ones(upper_n) * 500.0)
            kda = config.get("kds_arms", np.ones(upper_n) * 5.0)
            at = np.clip(arm_cmd, uql, uqh) if sc else arm_cmd
            arm_tau = pd_control(at, d.qpos[up_qadr], kpa, np.zeros(upper_n), d.qvel[up_vadr], kda)
            d.ctrl[leg_count:h12] = np.clip(np.nan_to_num(arm_tau), -max_tau, max_tau)
        if d.ctrl.shape[0] > h12:
            gr = m.actuator_ctrlrange[h12:, :]; d.ctrl[h12:] = 0.5 * (gr[:, 0] + gr[:, 1])

        mujoco.mj_step(m, d)

        # --- base drift / fall detection ---
        pelvis = d.qpos[:3]; Rp = quat2R(d.qpos[3:7]); ee_w = d.xpos[manip_eid]
        e_base = float(np.linalg.norm(pelvis[:2] - p0[:2]))
        tilt = float(np.degrees(np.arccos(np.clip(-get_gravity_orientation(d.qpos[3:7])[2], -1, 1))))
        ssq_base += e_base ** 2; max_tilt = max(max_tilt, tilt); nacc += 1
        if d.qpos[2] < HEIGHT_THRESHOLD or tilt > TILT_DEG_THRESHOLD:
            fell = True

        if renderer is not None and step % render_stride == 0:
            renderer.update_scene(d, camera=cam)
            if last_cmd_world is not None:
                if last_dist_world is not None:
                    add_error_line(renderer.scene, last_dist_world, last_cmd_world,
                                   rgba=(1.0, 0.25, 1.0, 0.9))
                    add_marker(renderer.scene, last_dist_world, radius=0.038,
                               rgba=(1.0, 0.2, 1.0, 0.95))
                add_marker(renderer.scene, last_cmd_world)            # green: fixed world target
            add_force_arrow(renderer.scene, ee_w, F)                  # red: payload weight
            l2 = (f"t={t:4.1f}s  load={kg_now:.1f}kg  pelvis drift={e_base*100:4.1f}cm  "
                  f"world err={e_ee*100:4.1f}cm  disturbance={e_ee_b*100:4.1f}cm")
            lines = [(label, (255, 255, 255)), (l2, (255, 255, 255)),
                     ("green=world target (held)   magenta=base disturbance IK rejects   red=load",
                      (180, 230, 160))]
            if fell:
                lines.append(("FELL", (255, 60, 60)))
            frames.append(overlay_text(renderer.render(), lines))

        if step % decim == 0:
            # commanded EE = fixed world target; dist EE = where the hand would be if the arm held
            # its nominal base-frame pose (the disturbance the IK rejects).
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
            traj["z"].append(float(np.linalg.norm(z)))   # FAME adapts; blind ~constant
            actor_obs = np.concatenate([np.concatenate(list(obs_hist)),
                                        np.flip(z_hist, 0).flatten()]).astype(np.float32)
            action = policy(torch.from_numpy(actor_obs).unsqueeze(0)).detach().numpy().squeeze()
            target_dof_pos = action * config["action_scale"] + config["default_angles"]

        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            _rem = dt - (time.time() - _t0)
            if _rem > 0:
                time.sleep(_rem)

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


# =============================================================================================
# Bimanual carry task (bi_manual_carry.yaml): both hands at fixed body-frame carry pose,
# torso joint sweeps a time schedule, payload drops at drop_at_s.
# =============================================================================================

def build_torso_schedule(keyframes):
    """keyframes = [(t_s, angle_rad), ...] -> callable t -> angle (smoothstep interp)."""
    times = np.array([k[0] for k in keyframes], dtype=float)
    angles = np.array([k[1] for k in keyframes], dtype=float)

    def at(t):
        if t <= times[0]:
            return float(angles[0])
        if t >= times[-1]:
            return float(angles[-1])
        i = int(np.searchsorted(times, t) - 1)
        span = times[i + 1] - times[i]
        u = smoothstep((t - times[i]) / span) if span > 1e-9 else 1.0
        return float(angles[i] + u * (angles[i + 1] - angles[i]))

    return at


def build_carry_payload(payload_kg, settle_s, load_ramp_s, drop_at_s, drop_ramp_s):
    """t -> kg per hand. Smoothstep ramp up from 0 starting at settle_s, then smoothstep drop
    to 0 starting at drop_at_s. Same kg(t) applied to both wrists."""
    def at(t):
        if t < settle_s:
            return 0.0
        up = smoothstep((t - settle_s) / max(load_ramp_s, 1e-6))
        down = 1.0 - smoothstep((t - drop_at_s) / max(drop_ramp_s, 1e-6))
        return float(payload_kg * up * down)
    return at


def _status_color(tilt_deg, fell):
    """Traffic-light color for the torso 'health' marker drawn on rendered frames.
    Green (stable) → yellow (tilt > half-threshold) → orange → red (fell)."""
    if fell:
        return (1.0, 0.1, 0.1, 1.0)
    if tilt_deg > TILT_DEG_THRESHOLD:
        return (1.0, 0.3, 0.1, 1.0)
    if tilt_deg > TILT_DEG_THRESHOLD * 0.5:
        return (1.0, 0.85, 0.1, 1.0)
    return (0.1, 1.0, 0.2, 1.0)


def run_carry(config, m, ids, policy, encoder, bounds, *, rm,
              torso_at, payload_at, total_s, left_arm_q, right_arm_q,
              no_encode, label="", seed=None, init_jitter=0.01, live_view=False,
              renderer=None, cam=None, render_stride=20):
    """One bimanual-carry episode.

    The arms are held at FIXED joint angles (the IK solution for the carry pose at the
    initial torso angle) — they sweep with the torso, not against it. The torso joint
    tracks ``torso_at(t)``; payload tracks ``payload_at(t)``; FAME's leg policy keeps the
    pelvis stable. Set ``no_encode=True`` to zero the encoder's force input (no-FAME).
    Set ``live_view=True`` to open a passive MuJoCo viewer at real-time speed.
    Pass ``renderer``+``cam`` to record offscreen frames (force arrows on hands +
    color-coded status ball above the torso); returned in ``metrics["frames"]``.
    """
    import collections
    import time as _time

    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    if seed is not None:
        rng = np.random.default_rng(int(seed))
        h12 = int(config.get("h12_ctrl_count", 27))
        jids = m.actuator_trnid[:h12, 0].astype(np.int32)
        qadr_init = m.jnt_qposadr[jids].astype(np.int32)
        d.qpos[qadr_init] += rng.normal(0.0, init_jitter, size=h12)
        mujoco.mj_forward(m, d)

    dt = config["simulation_dt"]; decim = config["control_decimation"]
    policy_joints = int(config.get("policy_num_joints", 27))
    h12 = int(config.get("h12_ctrl_count", 27))
    leg_count = config["num_actions"]; upper_n = h12 - leg_count
    jids = m.actuator_trnid[:h12, 0].astype(np.int32)
    qadr = m.jnt_qposadr[jids].astype(np.int32); vadr = m.jnt_dofadr[jids].astype(np.int32)
    leg_qadr, leg_vadr = qadr[:leg_count], vadr[:leg_count]
    up_qadr, up_vadr = qadr[leg_count:h12], vadr[leg_count:h12]
    lwid, rwid = ids["left_wrist"], ids["right_wrist"]

    p0 = d.qpos[:3].copy()  # nominal pelvis position for drift measurement
    action = np.zeros(leg_count, dtype=np.float32)
    target_dof_pos = config["default_angles"].copy()
    cmd = config["cmd_init"].copy(); height_cmd = float(config["height_cmd"])

    qj = d.qpos[qadr]; dqj = d.qvel[vadr]
    sobs, _ = compute_observation(d, config, action, cmd, height_cmd, policy_joints, qj=qj, dqj=dqj)
    obs_hist = collections.deque([sobs.copy()] * config["obs_history_len"],
                                  maxlen=config["obs_history_len"])
    z_hist = np.zeros((3, RMA_LATENT_DIM), dtype=np.float32)
    g = np.array([0.0, 0.0, -9.81]); max_tau = 200.0
    n_steps = int(total_s / dt)
    sc = bounds["safety_clip"]
    lql, lqh = bounds["leg_q_low"], bounds["leg_q_high"]
    uql, uqh = bounds["upper_q_low"], bounds["upper_q_high"]

    ssq_base = 0.0; max_tilt = 0.0; nacc = 0; fell = False
    traj = {"t": [], "bpos": [], "bquat": [], "torso_cmd": [], "torso_actual": [],
            "load": [], "z": []}

    viewer = None
    if live_view:
        import mujoco.viewer as _mjv
        viewer = _mjv.launch_passive(m, d)
        print(f"[carry] live viewer open — close window to abort  ({label})")

    # End-effector body ids for force-arrow rendering (red arrows = payload weight)
    # and torso body id for the status-ball marker.
    leid = ids["left_ee"]; reid = ids["right_ee"]
    torso_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    frames = [] if renderer is not None else None

    for step in range(n_steps):
        _t0 = _time.time()
        t = step * dt

        # Payload — same force on both wrists
        kg_now = float(payload_at(t))
        F = kg_now * g
        d.xfrc_applied[:] = 0
        d.xfrc_applied[lwid, :3] = F
        d.xfrc_applied[rwid, :3] = F
        ef_left = F.astype(np.float32)
        ef_right = F.astype(np.float32)

        # Build the 15-dof upper command: [torso(t), L_arm(fixed), R_arm(fixed)]
        arm_cmd = np.zeros(upper_n, dtype=np.float32)
        arm_cmd[0] = torso_at(t)
        arm_cmd[ARM_SLICE["left"]] = left_arm_q
        arm_cmd[ARM_SLICE["right"]] = right_arm_q

        # PD — legs from policy, upper from arm_cmd
        if sc:
            target_dof_pos = np.clip(target_dof_pos, lql, lqh)
        leg_tau = pd_control(target_dof_pos, d.qpos[leg_qadr], config["kps"],
                             np.zeros_like(config["kps"]), d.qvel[leg_vadr], config["kds"])
        d.ctrl[:leg_count] = np.clip(np.nan_to_num(leg_tau), -max_tau, max_tau)
        if upper_n > 0:
            kpa = config.get("kps_arms", np.ones(upper_n) * 500.0)
            kda = config.get("kds_arms", np.ones(upper_n) * 5.0)
            at_clip = np.clip(arm_cmd, uql, uqh) if sc else arm_cmd
            arm_tau = pd_control(at_clip, d.qpos[up_qadr], kpa,
                                 np.zeros(upper_n), d.qvel[up_vadr], kda)
            d.ctrl[leg_count:h12] = np.clip(np.nan_to_num(arm_tau), -max_tau, max_tau)
        if d.ctrl.shape[0] > h12:
            gr = m.actuator_ctrlrange[h12:, :]; d.ctrl[h12:] = 0.5 * (gr[:, 0] + gr[:, 1])

        mujoco.mj_step(m, d)

        # Metrics
        pelvis = d.qpos[:3]; pelvis_q = d.qpos[3:7]
        e_base = float(np.linalg.norm(pelvis[:2] - p0[:2]))
        tilt = float(np.degrees(np.arccos(
            np.clip(-get_gravity_orientation(pelvis_q)[2], -1, 1))))
        ssq_base += e_base ** 2; max_tilt = max(max_tilt, tilt); nacc += 1
        if pelvis[2] < HEIGHT_THRESHOLD or tilt > TILT_DEG_THRESHOLD:
            fell = True

        if step % decim == 0:
            traj["t"].append(t)
            traj["bpos"].append(pelvis.copy())
            traj["bquat"].append(pelvis_q.copy())
            traj["torso_cmd"].append(float(arm_cmd[0]))
            traj["torso_actual"].append(float(d.qpos[up_qadr[0]]))
            traj["load"].append(kg_now)

            qj = d.qpos[qadr]; dqj = d.qvel[vadr]
            sobs, _ = compute_observation(d, config, action, cmd, height_cmd, policy_joints,
                                          qj=qj, dqj=dqj)
            obs_hist.append(sobs)
            e_t = build_et_mujoco(d.qpos,
                                  np.zeros(3) if no_encode else ef_left,
                                  np.zeros(3) if no_encode else ef_right,
                                  leg_count, policy_joints, qj)
            with torch.no_grad():
                z = encoder(torch.from_numpy(e_t).unsqueeze(0).float()).numpy().squeeze()
            z_hist[1:] = z_hist[:-1]; z_hist[0] = z
            traj["z"].append(float(np.linalg.norm(z)))

            actor_obs = np.concatenate([np.concatenate(list(obs_hist)),
                                        np.flip(z_hist, 0).flatten()]).astype(np.float32)
            action = policy(torch.from_numpy(actor_obs).unsqueeze(0)).detach().numpy().squeeze()
            target_dof_pos = action * config["action_scale"] + config["default_angles"]

        # Offscreen render path (for sweep_video). Same scene markers as the live view:
        # force arrows on both hands + a color-coded status ball above the torso.
        if renderer is not None and step % render_stride == 0:
            renderer.update_scene(d, camera=cam)
            add_force_arrow(renderer.scene, d.xpos[leid], F)
            add_force_arrow(renderer.scene, d.xpos[reid], F)
            ball_pos = d.xpos[torso_bid] + np.array([0.0, 0.0, 0.65])
            add_marker(renderer.scene, ball_pos, radius=0.06,
                       rgba=_status_color(tilt, fell))
            l2 = f"t={t:4.1f}s  load={kg_now:.1f}kg/hand  tilt={tilt:4.1f}°"
            lines = [(label, (255, 255, 255)), (l2, (255, 255, 255))]
            if fell:
                lines.append(("FELL", (255, 60, 60)))
            frames.append(overlay_text(renderer.render(), lines))

        if viewer is not None:
            if not viewer.is_running():
                break
            # Draw payload-force arrows on both hands. Reset the user scene each tick so
            # arrows don't accumulate; the helper no-ops if |F| ≈ 0 (after the drop).
            viewer.user_scn.ngeom = 0
            add_force_arrow(viewer.user_scn, d.xpos[leid], F)
            add_force_arrow(viewer.user_scn, d.xpos[reid], F)
            ball_pos = d.xpos[torso_bid] + np.array([0.0, 0.0, 0.65])
            add_marker(viewer.user_scn, ball_pos, radius=0.06,
                       rgba=_status_color(tilt, fell))
            viewer.sync()
            _rem = dt - (_time.time() - _t0)
            if _rem > 0:
                _time.sleep(_rem)

    if viewer is not None:
        viewer.close()
    for k in list(traj):
        traj[k] = np.array(traj[k])
    return {
        "base_rmse": float(np.sqrt(ssq_base / max(1, nacc))),
        "tilt_max": max_tilt,
        "fell": int(fell),
        "frames": frames,
        "traj": traj,
    }


def run_bimanual_carry(args, mc, task, config, m, ids, policy, encoder, bounds,
                       rm, arm_down, payload, figdir):
    """Top-level dispatch for the bimanual carry task. Solves bimanual IK once at the
    initial torso, runs FAME and no-FAME conditions back-to-back, saves the comparison plot.
    """
    torso_at = build_torso_schedule(task["torso_schedule"])
    torso_t0 = float(task["torso_schedule"][0][1])
    total_s = float(task.get("total_s", 13.0))
    drop_at_s = float(task.get("drop_at_s", total_s))
    drop_ramp_s = float(mc.get("drop_ramp_s", 0.15))
    load_ramp_s = float(mc.get("load_ramp_s", 0.8))
    settle_s = 1.0
    payload_at = build_carry_payload(payload, settle_s, load_ramp_s, drop_at_s, drop_ramp_s)

    # Bimanual IK: solve each arm's carry pose independently (kinematic chains are disjoint)
    carry_pose = task["carry_pose"]
    wps_left = [{"name": "carry", "xyz": carry_pose["left"]}]
    wps_right = [{"name": "carry", "xyz": carry_pose["right"]}]
    oc = float(mc.get("orientation_cost", 0.0)) if mc.get("track_orientation") else 0.0
    sols_l, resid_l = solve_arm_waypoints(rm, "left", arm_down, torso_t0, wps_left,
                                          orientation_cost=oc)
    sols_r, resid_r = solve_arm_waypoints(rm, "right", arm_down, torso_t0, wps_right,
                                          orientation_cost=oc)
    left_arm_q = sols_l[0].astype(np.float32)
    right_arm_q = sols_r[0].astype(np.float32)

    print(f"[IK] bimanual_carry payload={payload}kg per hand  drop@{drop_at_s}s  total={total_s}s")
    print(f"   left  carry xyz={carry_pose['left']}  IK residual={resid_l[0] * 1000:5.1f}mm"
          f"{'' if resid_l[0] < 0.01 else '  <-- UNREACHABLE'}")
    print(f"   right carry xyz={carry_pose['right']} IK residual={resid_r[0] * 1000:5.1f}mm"
          f"{'' if resid_r[0] < 0.01 else '  <-- UNREACHABLE'}")
    print(f"   torso schedule: {task['torso_schedule']}")

    # --- Live view mode: run ONE condition with real-time MuJoCo viewer; no plot. ---
    if args.view:
        no_enc = (args.view_cond == "no-FAME")
        print(f"\n[live view] {args.view_cond}  (close the viewer window to stop)")
        mt = run_carry(config, m, ids, policy, encoder, bounds, rm=rm,
                       torso_at=torso_at, payload_at=payload_at, total_s=total_s,
                       left_arm_q=left_arm_q, right_arm_q=right_arm_q,
                       no_encode=no_enc, label=f"{args.view_cond} (bimanual_carry)",
                       live_view=True)
        print(f"  {args.view_cond:8s} pelvis drift rmse={mt['base_rmse'] * 100:5.2f} cm   "
              f"max tilt={mt['tilt_max']:5.1f}°   fell={mt['fell']}")
        return

    # --- Side-by-side payload-sweep video: FAME | no-FAME stepping through payloads. ---
    if args.sweep_video:
        viddir = os.path.join(_REPO_ROOT, "simulation_exp/videos")
        os.makedirs(viddir, exist_ok=True)
        payloads_per_hand = list(range(3, 11))   # 3-10 kg/hand
        renderer = mujoco.Renderer(m, height=args.vid_h, width=args.vid_w)
        cam = make_manip_camera()
        all_sbs = []
        print(f"\n[bimanual sweep_video] payloads {payloads_per_hand} kg/hand "
              f"— FAME (left) | no-FAME (right)")
        for kg_per_hand in payloads_per_hand:
            payload_at_kg = build_carry_payload(float(kg_per_hand), settle_s,
                                                load_ramp_s, drop_at_s, drop_ramp_s)
            clips = {}
            for lab, no_enc in [("FAME", False), ("no-FAME", True)]:
                mt = run_carry(config, m, ids, policy, encoder, bounds, rm=rm,
                               torso_at=torso_at, payload_at=payload_at_kg,
                               total_s=total_s,
                               left_arm_q=left_arm_q, right_arm_q=right_arm_q,
                               no_encode=no_enc,
                               renderer=renderer, cam=cam,
                               render_stride=args.render_stride,
                               label=f"{lab}  {kg_per_hand}kg/hand")
                clips[lab] = mt["frames"]
                print(f"  {kg_per_hand}kg/hand {lab:8s} "
                      f"drift={mt['base_rmse']*100:5.1f}cm  "
                      f"tilt={mt['tilt_max']:5.1f}°  fell={mt['fell']}")
            all_sbs.extend(sidebyside_frames(clips["FAME"], clips["no-FAME"]))
        renderer.close()
        out_path = os.path.join(viddir, "bimanual_carry_sweep_sidebyside.mp4")
        save_mp4(out_path, all_sbs, fps=args.fps)
        print(f"saved video -> {out_path}  ({len(all_sbs)} frames @ {args.fps} fps)")
        return

    # --- Payload sweep: payloads × seeds × {FAME, no-FAME}; aggregate + plot. ---
    if args.sweep:
        payloads_per_hand = list(range(1, 11))   # 1-10 kg/hand; encoder clipped to 30 N (≈3 kg)
        n_seeds = max(1, int(args.seeds))
        results = {"FAME": [[] for _ in payloads_per_hand],
                   "no-FAME": [[] for _ in payloads_per_hand]}
        print(f"\n[bimanual sweep] payloads {payloads_per_hand} kg/hand × {n_seeds} seed(s) "
              f"(init joint jitter std={args.init_jitter} rad)")
        for kg_idx, kg_per_hand in enumerate(payloads_per_hand):
            payload_at_kg = build_carry_payload(float(kg_per_hand), settle_s,
                                                load_ramp_s, drop_at_s, drop_ramp_s)
            for s in range(n_seeds):
                seed = s if n_seeds > 1 else None
                for lab, no_enc in [("FAME", False), ("no-FAME", True)]:
                    mt = run_carry(config, m, ids, policy, encoder, bounds, rm=rm,
                                   torso_at=torso_at, payload_at=payload_at_kg,
                                   total_s=total_s,
                                   left_arm_q=left_arm_q, right_arm_q=right_arm_q,
                                   no_encode=no_enc, seed=seed,
                                   init_jitter=args.init_jitter,
                                   label=f"{lab} ({kg_per_hand}kg/hand, seed={s})")
                    results[lab][kg_idx].append(mt)
                    print(f"  {kg_per_hand}kg/hand seed={s} {lab:8s} "
                          f"drift={mt['base_rmse']*100:5.2f}cm  "
                          f"tilt={mt['tilt_max']:5.1f}°  fell={mt['fell']}")
        from utils import plot_carry_summary_multiseed
        out_path = os.path.join(figdir, "bimanual_carry_summary.png")
        plot_carry_summary_multiseed(results, payloads_per_hand, out_path)
        return

    trajs = {}
    for lab, no_enc in [("FAME", False), ("no-FAME", True)]:
        mt = run_carry(config, m, ids, policy, encoder, bounds, rm=rm,
                       torso_at=torso_at, payload_at=payload_at, total_s=total_s,
                       left_arm_q=left_arm_q, right_arm_q=right_arm_q,
                       no_encode=no_enc, label=f"{lab} (bimanual_carry)")
        trajs[lab] = mt["traj"]
        print(f"  {lab:8s} pelvis drift rmse={mt['base_rmse'] * 100:5.2f} cm   "
              f"max tilt={mt['tilt_max']:5.1f}°   fell={mt['fell']}")

    from utils import plot_carry_compare
    out_path = os.path.join(figdir, "bimanual_carry_compare.png")
    plot_carry_compare(trajs, drop_at_s, task["torso_schedule"], out_path)


def main():
    p = argparse.ArgumentParser(description="Single-arm IK pick-place + FAME: world-EE decomposition demo")
    p.add_argument("--config", default=os.path.join(_SCRIPT_DIR, "h1_2_rma_arm_magpie_fame.yaml"))
    p.add_argument("--manip_yaml", default=None,
                   help="YAML path; auto-picked from --task if omitted "
                        "(single_arm_manip.yaml for *_hand_manip, bi_manual_carry.yaml for bimanual_carry).")
    p.add_argument("--task", default="right_hand_manip",
                   choices=("right_hand_manip", "left_hand_manip", "bimanual_carry"))
    p.add_argument("--payload_kg", type=float, default=None, help="Override the YAML payload")
    p.add_argument("--view", action="store_true", help="Open a live MuJoCo viewer (one condition, real-time)")
    p.add_argument("--view_cond", default="FAME", choices=("FAME", "no-FAME"))
    p.add_argument("--sweep", action="store_true",
                   help="Sweep payload 1-3kg; save commanded-vs-actual trajectory + summary figures")
    p.add_argument("--sweep_video", action="store_true",
                   help="Render ONE side-by-side video stepping through 1-3kg")
    p.add_argument("--adapt", action="store_true",
                   help="On-the-fly payload changes (steps in 0-3kg) while moving: FAME vs no-FAME")
    p.add_argument("--plot_payload", type=int, default=4, help="Payload (kg) for the trajectory plot")
    p.add_argument("--seeds", type=int, default=1,
                   help="Number of seeds (initial-joint-jitter trials) per payload × condition "
                        "in --sweep mode. >1 enables mean ± std error bars.")
    p.add_argument("--init_jitter", type=float, default=0.01,
                   help="Std (rad) of the per-joint Gaussian noise injected at t=0 when --seeds > 1.")
    p.add_argument("--video", default=None, help="path prefix for FAME/no-FAME/side-by-side mp4s")
    p.add_argument("--vid_w", type=int, default=640); p.add_argument("--vid_h", type=int, default=480)
    p.add_argument("--fps", type=int, default=25); p.add_argument("--render_stride", type=int, default=20)
    args = p.parse_args()

    # Auto-pick the yaml based on --task if not provided
    if args.manip_yaml is None:
        if args.task == "bimanual_carry":
            args.manip_yaml = os.path.join(_SCRIPT_DIR, "bi_manual_carry.yaml")
        else:
            args.manip_yaml = os.path.join(_SCRIPT_DIR, "single_arm_manip.yaml")

    mc = yaml.safe_load(open(args.manip_yaml))
    task = mc[args.task]; side = task["manip"]
    arm_down = np.asarray(mc["arm_down"], dtype=np.float32); torso = float(mc.get("torso", -0.35))
    payload = args.payload_kg if args.payload_kg is not None else float(task.get("payload_kg", 1.0))
    settle_s, seg_s, hold_s, ramp_s = (float(mc.get(k, v)) for k, v in
                                       [("load_ramp_s", 0.8), ("seg_time_s", 1.5),
                                        ("hold_s", 0.8), ("load_ramp_s", 0.8)])
    settle_s = 1.0
    oc = float(mc.get("orientation_cost", 0.0)) if mc.get("track_orientation") else 0.0

    # --- config / model / policy / encoder / safety ---
    config = load_config(args.config); cdir = os.path.dirname(os.path.abspath(args.config))
    for k in ("policy_path", "xml_path", "encoder_path"):
        if config.get(k) and not os.path.isabs(config[k]):
            config[k] = os.path.normpath(os.path.join(cdir, config[k]))
    config["_arm_down"] = arm_down
    m = mujoco.MjModel.from_xml_path(config["xml_path"]); m.opt.timestep = config["simulation_dt"]
    ids = {"left_wrist":  mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_roll_link"),
           "right_wrist": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link"),
           "left_ee":     mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link"),
           "right_ee":    mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")}
    policy = torch.jit.load(config["policy_path"]); policy.eval()
    encoder = EnvFactorEncoder(EnvFactorEncoderCfg())
    encoder.load_state_dict(torch.load(config["encoder_path"], map_location="cpu", weights_only=True))
    encoder.eval()
    sc = bool(config.get("safety_clip", True))
    bounds = dict(safety_clip=sc, leg_q_low=None, leg_q_high=None, upper_q_low=None, upper_q_high=None)
    if sc:
        lc = config["num_actions"]; hc = int(config.get("h12_ctrl_count", 27))
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(j))
                 for j in m.actuator_trnid[:hc, 0].astype(int)]
        qlo, qhi = load_safety_q_clip(names, config.get("safety_config"))
        bounds.update(leg_q_low=qlo[:lc], leg_q_high=qhi[:lc],
                      upper_q_low=qlo[lc:hc], upper_q_high=qhi[lc:hc])

    # Pinocchio model — used by every task variant (single-arm IK or bimanual carry)
    rm = RobotModel(os.path.join(_REPO_ROOT, "h1_2/h1_2_handless.urdf"), handless=True)

    figdir = os.path.join(_REPO_ROOT, "simulation_exp/figures"); os.makedirs(figdir, exist_ok=True)
    viddir = os.path.join(_REPO_ROOT, "simulation_exp/videos")

    # =============================================================================================
    # bimanual_carry dispatch: separate from single-arm waypoint logic (different yaml/run loop)
    # =============================================================================================
    if args.task == "bimanual_carry":
        run_bimanual_carry(args, mc, task, config, m, ids, policy, encoder, bounds,
                           rm, arm_down, payload, figdir)
        return

    # --- IK: arm joint waypoints for the EE targets (single-arm path) ---
    sols, resid = solve_arm_waypoints(rm, side, arm_down, torso, task["waypoints"], orientation_cost=oc)
    print(f"[IK] task={args.task} side={side} payload={payload}kg")
    for wp, r in zip(task["waypoints"], resid):
        flag = "" if r < 0.01 else "  <-- UNREACHABLE (residual high)"
        print(f"   {wp['name']:6s} xyz={wp['xyz']}  IK residual={r*1000:5.1f}mm{flag}")
    xyz_at, total = build_xyz_schedule(task["waypoints"], settle_s, seg_s, hold_s)

    if args.view:
        print(f"\n[live view] {args.task} — {args.view_cond}  (close the window to stop)")
        run_manip(config, m, ids, policy, encoder, bounds, side=side, rm=rm, xyz_at=xyz_at, total_s=total,
                  payload_kg=payload, settle_s=settle_s, load_ramp_s=ramp_s, torso=torso,
                  no_encode=(args.view_cond == "no-FAME"), live_view=True,
                  label=f"{args.view_cond} ({args.task}, {payload}kg)")
        return

    if args.sweep:
        payloads = list(range(1, 11))   # 1-10 kg; encoder is clipped to 30 N (≈3 kg)
        n_seeds = max(1, int(args.seeds))
        # results[lab][payload_idx] = [metric_dict for each seed]
        results = {"FAME": [[] for _ in payloads], "no-FAME": [[] for _ in payloads]}
        trajs = {}  # only saved for seed 0 at plot_payload
        print(f"\n[sweep] {args.task}: payloads {payloads} kg × {n_seeds} seed(s) "
              f"(init joint jitter std={args.init_jitter} rad)")
        for kg_idx, kg in enumerate(payloads):
            for s in range(n_seeds):
                seed = s if n_seeds > 1 else None
                for lab, no_enc in [("FAME", False), ("no-FAME", True)]:
                    mt, _ = run_manip(config, m, ids, policy, encoder, bounds, side=side, rm=rm,
                                      xyz_at=xyz_at, total_s=total, payload_kg=float(kg),
                                      settle_s=settle_s, load_ramp_s=ramp_s, torso=torso,
                                      no_encode=no_enc, seed=seed, init_jitter=args.init_jitter)
                    results[lab][kg_idx].append(mt)
                    if s == 0 and kg == args.plot_payload:
                        trajs[lab] = mt["traj"]
                    print(f"  {kg}kg seed={s} {lab:8s} e^W_ee={mt['ee_rmse']*100:5.1f}cm  "
                          f"e^W_base={mt['ee_b_rmse']*100:4.1f}cm  "
                          f"base={mt['base_rmse']*100:4.1f}cm  fell={mt['fell']}")
        if "FAME" in trajs and "no-FAME" in trajs:
            plot_traj(trajs, args.plot_payload, args.task,
                      os.path.join(figdir, f"manip_{side}_traj_{args.plot_payload}kg.png"))
        if n_seeds > 1:
            from utils import plot_summary_multiseed
            plot_summary_multiseed(results, payloads, args.task,
                                   os.path.join(figdir, f"manip_{side}_summary.png"))
        else:
            # Flatten the [[m]] nesting for the single-seed plot_summary.
            flat = {lab: [seeds_list[0] for seeds_list in results[lab]] for lab in results}
            plot_summary(flat, payloads, args.task,
                         os.path.join(figdir, f"manip_{side}_summary.png"))
        return

    if args.sweep_video:
        payloads = list(range(1, 11))   # 1-10 kg; encoder is clipped to 30 N (≈3 kg)
        renderer = mujoco.Renderer(m, height=args.vid_h, width=args.vid_w)
        cam = make_manip_camera()
        all_sbs = []
        print(f"\n[sweep-video] {args.task}: stepping payloads {payloads} kg (no-FAME | FAME)")
        for kg in payloads:
            clips = {}; met = {}
            for lab, no_enc in [("FAME", False), ("no-FAME", True)]:
                mt, frames = run_manip(config, m, ids, policy, encoder, bounds, side=side, rm=rm,
                                       xyz_at=xyz_at, total_s=total, payload_kg=float(kg),
                                       settle_s=settle_s, load_ramp_s=ramp_s, torso=torso,
                                       no_encode=no_enc, renderer=renderer, cam=cam,
                                       render_stride=args.render_stride,
                                       label=f"{lab}  ({args.task}, {kg}kg)")
                clips[lab] = frames; met[lab] = mt
            print(f"  {kg}kg  e^W_ee FAME={met['FAME']['ee_rmse']*100:4.1f}cm "
                  f"no-FAME={met['no-FAME']['ee_rmse']*100:4.1f}cm  | "
                  f"pelvis drift FAME={met['FAME']['base_rmse']*100:4.1f}cm "
                  f"no-FAME={met['no-FAME']['base_rmse']*100:4.1f}cm")
            all_sbs.extend(sidebyside_frames(clips["no-FAME"], clips["FAME"]))
        renderer.close()
        out = args.video or os.path.join(viddir, f"manip_{side}_world_sweep")
        save_mp4(f"{out}_sidebyside.mp4", all_sbs, fps=args.fps)
        print(f"saved {out}_sidebyside.mp4  (1-3kg sweep, no-FAME | FAME)")
        return

    if args.adapt:
        payload_at, steps = build_payload_profile(settle_s)
        renderer = mujoco.Renderer(m, height=args.vid_h, width=args.vid_w)
        cam = make_manip_camera()
        trajs = {}; clips = {}
        print(f"\n[adapt] {args.task}: ON-THE-FLY payload steps at t={[round(s, 1) for s in steps]}s")
        for lab, no_enc in [("FAME", False), ("no-FAME", True)]:
            mt, frames = run_manip(config, m, ids, policy, encoder, bounds, side=side, rm=rm,
                                   xyz_at=xyz_at, total_s=total, payload_kg=0.0,
                                   payload_at=payload_at, settle_s=settle_s, load_ramp_s=ramp_s,
                                   torso=torso, no_encode=no_enc, renderer=renderer, cam=cam,
                                   render_stride=args.render_stride,
                                   label=f"{lab}  on-the-fly load  ({args.task})")
            trajs[lab] = mt["traj"]; clips[lab] = frames
            print(f"  {lab:8s} world EE: rmse={mt['ee_rmse']*100:4.1f}cm peak={mt['ee_max']*100:5.1f}cm | "
                  f"pelvis drift rmse={mt['base_rmse']*100:4.1f}cm tilt={mt['tilt_max']:4.1f} fell={mt['fell']}")
        renderer.close()
        plot_adapt(trajs, steps, args.task, os.path.join(figdir, f"manip_{side}_adapt.png"))
        out = args.video or os.path.join(viddir, f"manip_{side}_adapt")
        save_sidebyside_mp4(f"{out}_sidebyside.mp4", clips["no-FAME"], clips["FAME"], fps=args.fps)
        print(f"saved {out}_sidebyside.mp4  (on-the-fly load, no-FAME | FAME)")
        return

    # default: single-payload FAME vs no-FAME, optional video
    renderer = cam = None
    if args.video:
        renderer = mujoco.Renderer(m, height=args.vid_h, width=args.vid_w)
        cam = make_manip_camera()

    clips = {}
    print(f"\n================ {args.task}: world-frame EE decomposition (FAME vs no-FAME) ================")
    for lab, no_enc in [("FAME", False), ("no-FAME", True)]:
        mt, frames = run_manip(config, m, ids, policy, encoder, bounds, side=side, rm=rm, xyz_at=xyz_at,
                               total_s=total, payload_kg=payload, settle_s=settle_s, load_ramp_s=ramp_s,
                               torso=torso, no_encode=no_enc, renderer=renderer, cam=cam,
                               render_stride=args.render_stride,
                               label=f"{lab}  ({args.task}, {payload}kg)")
        clips[lab] = frames
        print(f"  {lab:8s} e^W_ee (world track): rmse={mt['ee_rmse']*100:5.1f}cm "
              f"max={mt['ee_max']*100:5.1f}cm | e^W_base (disturb)={mt['ee_b_rmse']*100:5.1f}cm  "
              f"pelvis drift={mt['base_rmse']*100:4.1f}cm tilt={mt['tilt_max']:4.1f} fell={mt['fell']}")
    print("=================================================================================================")

    if args.video:
        for lab, key in [("no-FAME", "no_fame"), ("FAME", "fame")]:
            save_mp4(f"{args.video}_{key}.mp4", clips[lab], fps=args.fps)
            print(f"saved {args.video}_{key}.mp4")
        save_sidebyside_mp4(f"{args.video}_sidebyside.mp4", clips["no-FAME"], clips["FAME"], fps=args.fps)
        renderer.close()
        print(f"saved {args.video}_sidebyside.mp4  (no-FAME | FAME)")


if __name__ == "__main__":
    main()
