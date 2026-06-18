"""
RMA / FAME evaluation sweep (simulation), with a STATIC (quasi-static) and a DYNAMIC mode,
plus side-by-side video recording of FAME vs no-FAME.

Conditions (paired, identical scenario; only what the ENCODER sees differs):
  - "no_fame" : e_t force entries zeroed              -> encoder blind to the load.
  - "fame"    : e_t carries the real force, clipped to the trained magnitude -> in-distribution
                privileged-force latent.

STATIC mode (default): a constant force is applied to each wrist while the arms are held at a
fixed pose. Metric = world-frame EE error (RMSE/max/steady) + base drift.

DYNAMIC mode (--dynamic): the arms actively raise/lower (shoulder-pitch sweep) while holding a
payload of mass m; the payload force is INERTIAL, F = m*(g - a_wrist) (gravity + the held
mass's reaction during accel/decel) -> genuinely non-quasi-static. Metric = base drift + falls,
swept over payload mass (kg). The EE-from-nominal error is not meaningful here (arms move on
purpose) so base drift is the headline.

Joint commands are clamped to the h12_safety_layer limits before PD control (like the live runner).

Usage (from repo root):
  python .../sweep_rma_ablation.py --N 150 --down_hemi                       # static EE-error sweep
  python .../sweep_rma_ablation.py --dynamic --N 80 --payload_max 3.0        # dynamic base-drift sweep
  python .../sweep_rma_ablation.py --dynamic --video simulation_exp/videos/dyn --payload_kg 2.0  # videos
"""

import sys
import os
import argparse
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import mujoco
import torch
from mujoco_deploy_h12_rma import (
    load_config,
    load_safety_q_clip,
    pd_control,
    compute_observation,
    build_et_mujoco,
    get_gravity_orientation,
    RMA_LATENT_DIM,
    HAND_FORCE_MAG_MAX,
)
from h12_adaptive_policy.RMA.rma_modules.env_factor_encoder import (
    EnvFactorEncoder,
    EnvFactorEncoderCfg,
)

HEIGHT_THRESHOLD = 0.55
TILT_DEG_THRESHOLD = 45.0
GRASP_TOL_M = 0.03
GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float64)
# arm-vector indices (15): [torso, L_shoulder_pitch, L_sh_roll, L_sh_yaw, L_elbow, L_wr_roll,
#                           L_wr_pitch, L_wr_yaw, R_shoulder_pitch, R_sh_roll, ...]
L_SHOULDER_PITCH, R_SHOULDER_PITCH = 1, 8

CONDITIONS = [("no_fame", True), ("fame", False)]
DISPLAY = {"no_fame": "no-FAME", "fame": "FAME"}
COLORS = {"no_fame": "#d95f0e", "fame": "#2c7fb8"}


def get_tilt_deg(quat):
    g = get_gravity_orientation(quat)
    return float(np.degrees(np.arccos(np.clip(-g[2], -1.0, 1.0))))


def sample_in_range(rng, n, mag_max, down_hemi):
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    if down_hemi:
        d[:, 2] = -np.abs(d[:, 2])
    mags = rng.uniform(0.0, mag_max, size=(n, 1))
    return (d * mags).astype(np.float32)


def wrist_lin_vel(m, d, body_id):
    res = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, body_id, res, 0)  # world frame
    return res[3:6].copy()


def overlay(frame, lines):
    import cv2
    f = np.ascontiguousarray(frame)
    y = 24
    for txt, col in lines:
        cv2.putText(f, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        y += 26
    return f


def run_one(config, m, left_f, right_f, duration_s, policy, encoder,
            left_wrist_id, right_wrist_id, apply_forces, no_encode,
            safety_clip=False, leg_q_low=None, leg_q_high=None, upper_q_low=None, upper_q_high=None,
            dynamic=False, payload_kg=0.0, arm_freq=0.5, arm_amp=1.2,
            pickplace=False, reach_pose=None, ee_ref_L=None, ee_ref_R=None, arms="both",
            renderer=None, cam=None, render_stride=20, vid_label=""):
    """One headless episode. Returns (metrics_dict, frames_or_None)."""
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    L0 = d.xpos[left_wrist_id].copy(); R0 = d.xpos[right_wrist_id].copy()
    if ee_ref_L is not None:  # pick-place: measure EE error vs the reach TARGET, not the t=0 pose
        L0 = np.asarray(ee_ref_L, dtype=float); R0 = np.asarray(ee_ref_R, dtype=float)
    base0_xy = d.qpos[:2].copy()

    decim = config["control_decimation"]
    dt = config["simulation_dt"]
    policy_joints = int(config.get("policy_num_joints", 27))
    h12_ctrl_count = int(config.get("h12_ctrl_count", policy_joints))
    h12_joint_ids = m.actuator_trnid[:h12_ctrl_count, 0].astype(np.int32)
    h12_qpos_adr = m.jnt_qposadr[h12_joint_ids].astype(np.int32)
    h12_qvel_adr = m.jnt_dofadr[h12_joint_ids].astype(np.int32)
    leg_count = config["num_actions"]
    leg_qpos_adr = h12_qpos_adr[:leg_count]; leg_qvel_adr = h12_qvel_adr[:leg_count]
    upper_qpos_adr = h12_qpos_adr[leg_count:h12_ctrl_count]; upper_qvel_adr = h12_qvel_adr[leg_count:h12_ctrl_count]

    left_f = np.asarray(left_f, dtype=np.float32); right_f = np.asarray(right_f, dtype=np.float32)
    upper_h12_count = h12_ctrl_count - leg_count
    arm_base = np.asarray(config.get("default_angles_arms", np.zeros(upper_h12_count, dtype=np.float32)),
                          dtype=np.float32).copy()
    if len(arm_base) < upper_h12_count:
        arm_base = np.zeros(upper_h12_count, dtype=np.float32)

    action = np.zeros(leg_count, dtype=np.float32)
    target_dof_pos = config["default_angles"].copy()
    cmd = config["cmd_init"].copy()
    height_cmd = float(config["height_cmd"])

    qj_h12 = d.qpos[h12_qpos_adr]; dqj_h12 = d.qvel[h12_qvel_adr]
    single_obs, _ = compute_observation(d, config, action, cmd, height_cmd, policy_joints, qj=qj_h12, dqj=dqj_h12)
    obs_history = collections.deque(maxlen=config["obs_history_len"])
    for _ in range(config["obs_history_len"]):
        obs_history.append(single_obs.copy())

    z_history = np.zeros((3, RMA_LATENT_DIM), dtype=np.float32)
    max_tau = 200.0
    n_steps = int(duration_s / dt)
    ss_start = max(0, n_steps - max(1, int(round(1.0 / dt))))

    sumsqL = sumsqR = sumsq_base = 0.0
    maxL = maxR = max_tilt = max_payload = 0.0
    ssL_sum = ssR_sum = 0.0
    n_acc = ss_n = 0
    fell = False
    prev_vL = prev_vR = None
    frames = [] if renderer is not None else None

    for step in range(n_steps):
        t = step * dt
        d.xfrc_applied[:] = 0
        if dynamic:
            # inertial payload: F = m*(g - a_wrist), a from world-velocity finite difference
            vL = wrist_lin_vel(m, d, left_wrist_id); vR = wrist_lin_vel(m, d, right_wrist_id)
            aL = (vL - prev_vL) / dt if prev_vL is not None else np.zeros(3)
            aR = (vR - prev_vR) / dt if prev_vR is not None else np.zeros(3)
            prev_vL, prev_vR = vL, vR
            FL = payload_kg * (GRAVITY - aL); FR = payload_kg * (GRAVITY - aR)
            FL = np.clip(FL, -120.0, 120.0); FR = np.clip(FR, -120.0, 120.0)
            d.xfrc_applied[left_wrist_id, :3] = FL
            d.xfrc_applied[right_wrist_id, :3] = FR
            max_payload = max(max_payload, float(np.linalg.norm(FL)), float(np.linalg.norm(FR)))
            ef_left_raw, ef_right_raw = FL.astype(np.float32), FR.astype(np.float32)
            # arm raise/lower trajectory (shoulder pitch sweep), smooth start at nominal
            phase = 0.5 - 0.5 * np.cos(2 * np.pi * arm_freq * t)
            arm_cmd = arm_base.copy()
            arm_cmd[L_SHOULDER_PITCH] -= arm_amp * phase
            arm_cmd[R_SHOULDER_PITCH] -= arm_amp * phase
        elif pickplace:
            # quasi-static pick-place: slowly reach to the target pose, ramp the load on, hold.
            # `arms` selects which arm(s) reach and carry the load: "left", "right", or "both".
            rp = np.clip((t / duration_s - 0.10) / 0.35, 0.0, 1.0); rp = rp * rp * (3 - 2 * rp)  # reach
            lr = np.clip((t / duration_s - 0.50) / 0.12, 0.0, 1.0); lr = lr * lr * (3 - 2 * lr)  # load ramp
            reach = np.asarray(reach_pose, dtype=np.float32)
            arm_cmd = arm_base.copy()
            if arms in ("left", "both"):
                arm_cmd[1:8] = arm_base[1:8] + rp * (reach[1:8] - arm_base[1:8])     # left arm joints
            if arms in ("right", "both"):
                arm_cmd[8:15] = arm_base[8:15] + rp * (reach[8:15] - arm_base[8:15])  # right arm joints
            F = (lr * payload_kg) * GRAVITY  # downward weight only (slow => negligible inertial)
            FL = F if arms in ("left", "both") else np.zeros(3)
            FR = F if arms in ("right", "both") else np.zeros(3)
            d.xfrc_applied[left_wrist_id, :3] = FL
            d.xfrc_applied[right_wrist_id, :3] = FR
            max_payload = max(max_payload, float(np.linalg.norm(F)))
            ef_left_raw, ef_right_raw = FL.astype(np.float32), FR.astype(np.float32)
        else:
            if apply_forces:
                d.xfrc_applied[left_wrist_id, :3] = left_f
                d.xfrc_applied[right_wrist_id, :3] = right_f
            ef_left_raw, ef_right_raw = left_f, right_f
            arm_cmd = arm_base

        if safety_clip:
            target_dof_pos = np.clip(target_dof_pos, leg_q_low, leg_q_high)
        leg_tau = pd_control(target_dof_pos, d.qpos[leg_qpos_adr], config["kps"],
                             np.zeros_like(config["kps"]), d.qvel[leg_qvel_adr], config["kds"])
        leg_tau = np.clip(np.nan_to_num(leg_tau, nan=0.0, posinf=0.0, neginf=0.0), -max_tau, max_tau)
        d.ctrl[:leg_count] = leg_tau

        if upper_h12_count > 0:
            kps_arm = config.get("kps_arms", np.ones(upper_h12_count, dtype=np.float32) * 500.0)
            kds_arm = config.get("kds_arms", np.ones(upper_h12_count, dtype=np.float32) * 5.0)
            arm_t = arm_cmd[:upper_h12_count]
            if safety_clip:
                arm_t = np.clip(arm_t, upper_q_low, upper_q_high)
            arm_tau = pd_control(arm_t, d.qpos[upper_qpos_adr], kps_arm,
                                 np.zeros(upper_h12_count), d.qvel[upper_qvel_adr], kds_arm)
            arm_tau = np.clip(np.nan_to_num(arm_tau, nan=0.0, posinf=0.0, neginf=0.0), -max_tau, max_tau)
            d.ctrl[leg_count:h12_ctrl_count] = arm_tau

        if d.ctrl.shape[0] > h12_ctrl_count:
            gr = m.actuator_ctrlrange[h12_ctrl_count:, :]
            d.ctrl[h12_ctrl_count:] = 0.5 * (gr[:, 0] + gr[:, 1])

        mujoco.mj_step(m, d)

        eL = float(np.linalg.norm(d.xpos[left_wrist_id] - L0))
        eR = float(np.linalg.norm(d.xpos[right_wrist_id] - R0))
        base_dxy = float(np.linalg.norm(d.qpos[:2] - base0_xy))
        tilt = get_tilt_deg(d.qpos[3:7])
        sumsqL += eL * eL; sumsqR += eR * eR; sumsq_base += base_dxy * base_dxy
        maxL = max(maxL, eL); maxR = max(maxR, eR); max_tilt = max(max_tilt, tilt)
        n_acc += 1
        if step >= ss_start:
            ssL_sum += eL; ssR_sum += eR; ss_n += 1
        if d.qpos[2] < HEIGHT_THRESHOLD or tilt > TILT_DEG_THRESHOLD:
            fell = True

        if renderer is not None and step % render_stride == 0:
            renderer.update_scene(d, camera=cam)
            l2 = f"t={t:4.2f}s  load={payload_kg:.1f}kg/hand  base drift={base_dxy*100:4.1f}cm"
            lines = [(vid_label, (255, 255, 255)), (l2, (255, 255, 255))]
            if fell:
                lines.append(("FELL", (255, 60, 60)))
            frames.append(overlay(renderer.render(), lines))

        if step % decim == 0:
            qj_h12 = d.qpos[h12_qpos_adr]; dqj_h12 = d.qvel[h12_qvel_adr]
            single_obs, _ = compute_observation(d, config, action, cmd, height_cmd, policy_joints, qj=qj_h12, dqj=dqj_h12)
            obs_history.append(single_obs)
            ef_left = np.zeros(3, dtype=np.float32) if no_encode else ef_left_raw
            ef_right = np.zeros(3, dtype=np.float32) if no_encode else ef_right_raw
            e_t = build_et_mujoco(d.qpos, ef_left, ef_right, leg_count, policy_joints, qj_h12)
            if encoder is not None:
                with torch.no_grad():
                    z_t = encoder(torch.from_numpy(e_t).unsqueeze(0).float()).numpy().squeeze()
            else:
                z_t = np.zeros(RMA_LATENT_DIM, dtype=np.float32)
            z_history[1:, :] = z_history[:-1, :].copy(); z_history[0, :] = z_t
            z_flat = np.flip(z_history, axis=0).flatten().astype(np.float32)
            actor_obs = np.concatenate([np.concatenate(list(obs_history), axis=0), z_flat], axis=0).astype(np.float32)
            action = policy(torch.from_numpy(actor_obs).unsqueeze(0)).detach().numpy().squeeze()
            target_dof_pos = action * config["action_scale"] + config["default_angles"]

    metrics = {
        "rmse_L": float(np.sqrt(sumsqL / n_acc)), "rmse_R": float(np.sqrt(sumsqR / n_acc)),
        "rmse": float(0.5 * (np.sqrt(sumsqL / n_acc) + np.sqrt(sumsqR / n_acc))),
        "max": max(maxL, maxR), "ss": 0.5 * (ssL_sum + ssR_sum) / max(1, ss_n),
        "base_rmse": float(np.sqrt(sumsq_base / n_acc)), "base_tilt_max": max_tilt,
        "max_payload_N": max_payload, "fell": int(fell),
    }
    return metrics, frames


def _setup(args):
    config_path = args.config if os.path.isabs(args.config) else os.path.join(_SCRIPT_DIR, args.config)
    config = load_config(config_path)
    cfg_dir = os.path.dirname(os.path.abspath(config_path))
    for key in ["policy_path", "xml_path", "encoder_path"]:
        if config.get(key) and not os.path.isabs(config[key]):
            config[key] = os.path.normpath(os.path.join(cfg_dir, config[key]))
    m = mujoco.MjModel.from_xml_path(config["xml_path"])
    m.opt.timestep = config["simulation_dt"]
    lwid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_roll_link")
    rwid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")
    policy = torch.jit.load(config["policy_path"]); policy.eval()
    encoder = EnvFactorEncoder(EnvFactorEncoderCfg())
    encoder.load_state_dict(torch.load(config["encoder_path"], map_location="cpu", weights_only=True)); encoder.eval()
    # safety clamp bounds
    safety_clip = bool(config.get("safety_clip", True))
    bounds = dict(safety_clip=safety_clip, leg_q_low=None, leg_q_high=None, upper_q_low=None, upper_q_high=None)
    if safety_clip:
        lc = config["num_actions"]; hc = int(config.get("h12_ctrl_count", config.get("policy_num_joints", 27)))
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in m.actuator_trnid[:hc, 0].astype(int)]
        qlo, qhi = load_safety_q_clip(names, config.get("safety_config"))
        bounds.update(leg_q_low=qlo[:lc], leg_q_high=qhi[:lc], upper_q_low=qlo[lc:hc], upper_q_high=qhi[lc:hc])
        print(f"[safety] target positions clamped via h12_safety_layer ({len(names)} joints)")
    return config, m, lwid, rwid, policy, encoder, bounds, (lwid >= 0 and rwid >= 0)


def make_camera():
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.85]; cam.distance = 3.4; cam.azimuth = 130.0; cam.elevation = -10.0
    return cam


def record_videos(args):
    import imageio
    config, m, lwid, rwid, policy, encoder, bounds, apply_forces = _setup(args)
    print(f"[video] dynamic arms (raise/lower {args.arm_freq}Hz, amp {args.arm_amp}rad), payload {args.payload_kg}kg")
    renderer = mujoco.Renderer(m, height=args.vid_h, width=args.vid_w)
    cam = make_camera()
    clips = {}
    for lab, no_enc in CONDITIONS:
        _, frames = run_one(config, m, np.zeros(3), np.zeros(3), args.duration, policy, encoder,
                            lwid, rwid, apply_forces, no_encode=no_enc, dynamic=True,
                            payload_kg=args.payload_kg, arm_freq=args.arm_freq, arm_amp=args.arm_amp,
                            renderer=renderer, cam=cam, render_stride=args.render_stride,
                            vid_label=DISPLAY[lab], **bounds)
        path = f"{args.video}_{lab}.mp4"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        imageio.mimwrite(path, frames, fps=args.fps, codec="libx264", quality=8)
        clips[lab] = frames
        print(f"saved {path}  ({len(frames)} frames)")
    # side-by-side
    n = min(len(clips["fame"]), len(clips["no_fame"]))
    div = np.full((clips["fame"][0].shape[0], 4, 3), 255, np.uint8)
    sbs = [np.concatenate([clips["no_fame"][i], div, clips["fame"][i]], axis=1) for i in range(n)]
    sbs_path = f"{args.video}_sidebyside.mp4"
    imageio.mimwrite(sbs_path, sbs, fps=args.fps, codec="libx264", quality=8)
    renderer.close()
    print(f"saved {sbs_path}  (no-FAME | FAME)")


def ee_targets(m, config, reach_pose, lwid, rwid):
    """World-frame wrist positions for the reach pose with the base at nominal stance
    (the 'target' the hand should reach if the base never moved)."""
    dref = mujoco.MjData(m)
    hc = int(config.get("h12_ctrl_count", config.get("policy_num_joints", 27)))
    jids = m.actuator_trnid[:hc, 0].astype(np.int32)
    qadr = m.jnt_qposadr[jids].astype(np.int32)
    lc = config["num_actions"]
    full = np.concatenate([np.asarray(config["default_angles"], dtype=float)[:lc],
                           np.asarray(reach_pose, dtype=float)])
    dref.qpos[qadr] = full
    mujoco.mj_forward(m, dref)
    return dref.xpos[lwid].copy(), dref.xpos[rwid].copy()


def run_pickplace(args):
    """One pick-place demo: reach to a target pose, ramp the load on, hold. FAME vs no-FAME.
    Headline = world-frame hand error vs the reach target during the hold (FAME keeps the base
    steady so the hand stays on target; no-FAME drifts)."""
    config, m, lwid, rwid, policy, encoder, bounds, apply_forces = _setup(args)
    presets = config.get("arm_pose_presets", {})
    if args.reach not in presets:
        raise SystemExit(f"--reach must be one of {list(presets)}")
    arm_base = np.asarray(config["default_angles_arms"], dtype=np.float32)
    reach_pose = np.asarray(presets[args.reach], dtype=np.float32).copy()
    reach_pose[0] = arm_base[0]  # keep the trained torso (waist) offset
    eL, eR = ee_targets(m, config, reach_pose, lwid, rwid)
    print(f"[pickplace] reach='{args.reach}'  payload={args.payload_kg}kg  "
          f"EE target L={np.round(eL,3)} R={np.round(eR,3)}")
    renderer = None; cam = None
    if args.video:
        renderer = mujoco.Renderer(m, height=args.vid_h, width=args.vid_w); cam = make_camera()
    clips = {}
    for lab, no_enc in CONDITIONS:
        mt, frames = run_one(config, m, np.zeros(3), np.zeros(3), args.duration, policy, encoder,
                             lwid, rwid, apply_forces, no_encode=no_enc, **bounds,
                             pickplace=True, reach_pose=reach_pose, payload_kg=args.payload_kg, arms=args.arms,
                             ee_ref_L=eL, ee_ref_R=eR, renderer=renderer, cam=cam,
                             render_stride=args.render_stride,
                             vid_label=f"{DISPLAY[lab]} ({args.arms} {args.reach}, {args.payload_kg}kg)")
        clips[lab] = (mt, frames)
        print(f"  {DISPLAY[lab]:8s} hand-vs-target(hold)={mt['ss']*100:5.1f}cm  "
              f"base drift={mt['base_rmse']*100:5.1f}cm  tilt_max={mt['base_tilt_max']:4.1f}deg  fell={mt['fell']}")
    if args.video:
        import imageio
        for lab in [c[0] for c in CONDITIONS]:
            frames = clips[lab][1]; path = f"{args.video}_{lab}.mp4"
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            imageio.mimwrite(path, frames, fps=args.fps, codec="libx264", quality=8)
            print(f"saved {path}  ({len(frames)} frames)")
        ff = clips["fame"][1]; nf = clips["no_fame"][1]; n = min(len(ff), len(nf))
        div = np.full((ff[0].shape[0], 4, 3), 255, np.uint8)
        sbs = [np.concatenate([nf[i], div, ff[i]], axis=1) for i in range(n)]
        sp = f"{args.video}_sidebyside.mp4"
        imageio.mimwrite(sp, sbs, fps=args.fps, codec="libx264", quality=8)
        renderer.close(); print(f"saved {sp}  (no-FAME | FAME)")


ENV_CONFIGS = [("left", "Left arm"), ("right", "Right arm"), ("both", "Bimanual")]


def run_envelope(args):
    """Pick-place envelope: for left-arm / right-arm / bimanual, sweep payload and record the
    base drift e_base^W (FAME's term) + falls, FAME vs no-FAME. Writes CSV and the figure."""
    config, m, lwid, rwid, policy, encoder, bounds, apply_forces = _setup(args)
    presets = config.get("arm_pose_presets", {})
    if args.reach not in presets:
        raise SystemExit(f"--reach must be one of {list(presets)}")
    arm_base = np.asarray(config["default_angles_arms"], dtype=np.float32)
    reach_pose = np.asarray(presets[args.reach], dtype=np.float32).copy(); reach_pose[0] = arm_base[0]
    eL, eR = ee_targets(m, config, reach_pose, lwid, rwid)
    payloads = np.round(np.arange(0.5, args.payload_max + 1e-6, 0.5), 2)
    print(f"[envelope] reach='{args.reach}'  payloads {payloads[0]}..{payloads[-1]}kg  configs={[c[0] for c in ENV_CONFIGS]}")
    rows = []
    for arms, _ in ENV_CONFIGS:
        for kg in payloads:
            line = f"  {arms:5s} {kg:4.1f}kg "
            for lab, no_enc in CONDITIONS:
                mt, _ = run_one(config, m, np.zeros(3), np.zeros(3), args.duration, policy, encoder,
                                lwid, rwid, apply_forces, no_encode=no_enc, **bounds,
                                pickplace=True, reach_pose=reach_pose, payload_kg=float(kg), arms=arms,
                                ee_ref_L=eL, ee_ref_R=eR)
                rows.append((arms, float(kg), DISPLAY[lab], mt["base_rmse"], mt["base_tilt_max"], mt["fell"]))
                line += f" {DISPLAY[lab]}={mt['base_rmse']*100:4.1f}cm{'(fell)' if mt['fell'] else '     '}"
            print(line)
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w") as f:
        f.write("arms,payload_kg,cond,base_rmse,tilt_max,fell\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    print(f"saved CSV -> {args.csv}")
    plot_envelope(args.csv, args.out)


def plot_envelope(csv_path, out_path):
    import csv as _csv
    from matplotlib.lines import Line2D
    rows = list(_csv.DictReader(open(csv_path)))
    TAU_CM = 10.0   # illustrative IK-compensable base-drift budget
    YCAP = 25.0
    styles = {"FAME": ("#2c7fb8", "o", "-"), "no-FAME": ("#d95f0e", "s", "--")}
    fig, axes = plt.subplots(1, len(ENV_CONFIGS), figsize=(5.0 * len(ENV_CONFIGS), 4.8), sharey=True)
    if len(ENV_CONFIGS) == 1:
        axes = [axes]
    for ax, (arms, title) in zip(axes, ENV_CONFIGS):
        ax.axhspan(0, TAU_CM, color="#2ca02c", alpha=0.10, zorder=0)
        for cond, (color, mk, ls) in styles.items():
            sel = sorted([r for r in rows if r["arms"] == arms and r["cond"] == cond],
                         key=lambda r: float(r["payload_kg"]))
            x = [float(r["payload_kg"]) for r in sel]
            y = [min(float(r["base_rmse"]) * 100, YCAP) for r in sel]
            fell = [int(float(r["fell"])) for r in sel]
            ax.plot(x, y, ls, color=color, marker=mk, ms=5, lw=2, label=cond, zorder=3)
            xf = [xi for xi, fi in zip(x, fell) if fi]
            ax.scatter(xf, [YCAP] * len(xf), marker="x", color="red", s=55, lw=2, zorder=4)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("payload per loaded hand (kg)")
        ax.grid(alpha=0.3); ax.set_ylim(0, YCAP + 2)
    axes[0].set_ylabel(r"base drift  $e^{W}_{base}$  (cm)")
    axes[0].text(0.03, 0.30, "IK-compensable\n(controller closes it)", transform=axes[0].transAxes,
                 fontsize=8.5, color="#2ca02c", va="top")
    h, l = axes[0].get_legend_handles_labels()
    h.append(Line2D([], [], marker="x", color="red", ls="none")); l.append("fell")
    axes[0].legend(h, l, loc="upper left", fontsize=9)
    fig.suptitle(
        r"World-frame EE error   $e^{W}_{ee} = e^{W}_{base} + e^{B}_{ee}$"
        r"       ($e^{W}_{base}$ = FAME's job · base drift,   $e^{B}_{ee}$ = upper-body controller's job)"
        "\nFAME keeps base drift small & compensable; without it the base leaves the IK-compensable region and falls",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=150)
    print(f"saved figure -> {out_path}")


def run_sweep(args):
    config, m, lwid, rwid, policy, encoder, bounds, apply_forces = _setup(args)
    rng = np.random.default_rng(args.seed)
    labels = [c[0] for c in CONDITIONS]
    res = {lab: [] for lab in labels}

    if args.dynamic:
        loads = rng.uniform(0.0, args.payload_max, size=args.N).astype(np.float32)  # payload kg
        print(f"[dynamic] payload ~ U(0,{args.payload_max}kg), arms raise/lower {args.arm_freq}Hz amp {args.arm_amp}rad; metric=base drift")
        xlabel, xunit, xscale = "payload (kg)", "kg", 1.0
    else:
        left_F = sample_in_range(rng, args.N, args.mag_max, args.down_hemi)
        right_F = sample_in_range(rng, args.N, args.mag_max, args.down_hemi)
        loads = np.linalg.norm(left_F, axis=1) + np.linalg.norm(right_F, axis=1)
        print(f"[static] per-hand |F| ~ U(0,{args.mag_max}N) {'down' if args.down_hemi else 'uniform'}; metric=EE world error")
        xlabel, xunit, xscale = "total applied load (N)", "N", 1.0

    for i in range(args.N):
        line = f"[{i+1}/{args.N}] load={loads[i]:5.2f}{xunit} "
        for lab, no_enc in CONDITIONS:
            kw = dict(dynamic=True, payload_kg=float(loads[i]), arm_freq=args.arm_freq, arm_amp=args.arm_amp) \
                if args.dynamic else dict(left_f=left_F[i], right_f=right_F[i])
            mt, _ = run_one(config, m, kw.pop("left_f", np.zeros(3)), kw.pop("right_f", np.zeros(3)),
                            args.duration, policy, encoder, lwid, rwid, apply_forces,
                            no_encode=no_enc, **bounds, **kw)
            res[lab].append(mt)
            key = "base_rmse" if args.dynamic else "rmse"
            line += f" {DISPLAY[lab]}={mt[key]*100:4.1f}cm{'(fell)' if mt['fell'] else ''}"
        if (i + 1) % 5 == 0 or i == 0 or i == args.N - 1:
            print(line)

    def col(lab, k):
        return np.array([r[k] for r in res[lab]])

    headline = "base_rmse" if args.dynamic else "rmse"
    title = "base drift" if args.dynamic else "EE world error"
    print(f"\n================ {title} — FAME vs no-FAME ================")
    for lab in labels:
        print(f"  {DISPLAY[lab]:8s} {headline}={col(lab,headline).mean()*100:5.2f}cm  "
              f"base={col(lab,'base_rmse').mean()*100:5.2f}cm  EE-rmse={col(lab,'rmse').mean()*100:5.2f}cm  "
              f"tilt_max={col(lab,'base_tilt_max').mean():4.1f}deg  fell={int(col(lab,'fell').sum())}/{args.N}")
    print("==============================================================")

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    keys = ["rmse", "max", "ss", "base_rmse", "base_tilt_max", "max_payload_N", "fell"]
    with open(args.csv, "w") as f:
        f.write("load,cond," + ",".join(keys) + "\n")
        for i in range(args.N):
            for lab in labels:
                r = res[lab][i]
                f.write(",".join(str(v) for v in (loads[i], DISPLAY[lab], *[r[k] for k in keys])) + "\n")
    print(f"saved CSV -> {args.csv}")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    lm = loads
    nb = 6
    edges = np.linspace(lm.min(), lm.max() + 1e-9, nb + 1); centers = 0.5 * (edges[:-1] + edges[1:])
    for lab in labels:
        y = col(lab, headline) * 100.0
        ax.scatter(lm, y, s=12, alpha=0.25, color=COLORS[lab])
        means = [np.nanmean(y[(lm >= edges[b]) & (lm < edges[b + 1])]) if ((lm >= edges[b]) & (lm < edges[b + 1])).any()
                 else np.nan for b in range(nb)]
        ax.plot(centers, means, "o-" if lab == "fame" else "s--", color=COLORS[lab], label=DISPLAY[lab])
    if not args.dynamic:
        ax.axhline(GRASP_TOL_M * 100, color="gray", ls=":", lw=1)
    ax.set_xlabel(xlabel); ax.set_ylabel(f"{title} (cm)")
    ax.set_title(f"{'DYNAMIC' if args.dynamic else 'STATIC'}: {title} vs load (FAME vs no-FAME)")
    ax.legend(); ax.grid(alpha=0.3)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out, dpi=150)
    print(f"saved figure -> {args.out}")


def main():
    p = argparse.ArgumentParser(description="FAME eval sweep: static (EE error) or dynamic (base drift), + videos")
    p.add_argument("--config", type=str, default=os.path.join(_SCRIPT_DIR, "h1_2_rma_arm_magpie_fame.yaml"))
    p.add_argument("--N", type=int, default=120)
    p.add_argument("--duration", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=0)
    # static sampling
    p.add_argument("--mag_max", type=float, default=HAND_FORCE_MAG_MAX)
    p.add_argument("--down_hemi", action="store_true")
    # dynamic
    p.add_argument("--dynamic", action="store_true", help="Arms raise/lower with an inertial payload (non-quasi-static)")
    p.add_argument("--payload_max", type=float, default=3.0, help="Dynamic sweep: payload ~ U(0, payload_max) kg")
    p.add_argument("--payload_kg", type=float, default=2.0, help="Fixed payload for --video")
    p.add_argument("--arm_freq", type=float, default=0.5, help="Arm raise/lower frequency (Hz)")
    p.add_argument("--arm_amp", type=float, default=1.2, help="Shoulder-pitch sweep amplitude (rad)")
    # pick-place (one demo): reach to a pose and hold under load
    p.add_argument("--pickplace", action="store_true", help="Quasi-static reach-and-hold-under-load demo (FAME vs no-FAME)")
    p.add_argument("--envelope", action="store_true", help="Pick-place envelope sweep: left/right/bimanual x payload, base-drift figure")
    p.add_argument("--arms", type=str, default="both", choices=("left", "right", "both"), help="Which arm(s) carry the load (pickplace demo)")
    p.add_argument("--reach", type=str, default="forward_extended", help="Arm preset to reach to (from arm_pose_presets)")
    p.add_argument("--replot", action="store_true", help="Envelope: re-plot from existing CSV without re-running sims")
    # video
    p.add_argument("--video", type=str, default=None, help="Path prefix; records FAME/no-FAME/side-by-side mp4s")
    p.add_argument("--vid_w", type=int, default=640); p.add_argument("--vid_h", type=int, default=480)
    p.add_argument("--fps", type=int, default=25); p.add_argument("--render_stride", type=int, default=20)
    # outputs
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    tag = "envelope" if args.envelope else ("dyn" if args.dynamic else "static")
    if args.csv is None:
        args.csv = os.path.join(_REPO_ROOT, f"simulation_exp/data/{tag}.csv")
    if args.out is None:
        args.out = os.path.join(_REPO_ROOT, f"simulation_exp/figures/{tag}.png")

    if args.envelope:
        plot_envelope(args.csv, args.out) if args.replot else run_envelope(args)
    elif args.pickplace:
        run_pickplace(args)
    elif args.video:
        record_videos(args)
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()
