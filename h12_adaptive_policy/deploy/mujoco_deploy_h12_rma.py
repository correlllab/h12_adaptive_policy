"""
Same as mujoco_deploy_h12.py with RMA on top:
- Hand-only 3D forces (left/right wrist); no torso. e_t = 15 upper-body + left_xyz(3) + right_xyz(3) = 21.
- Apply forces to left_wrist_roll_link and right_wrist_roll_link; build e_t and run encoder -> z_t.
- Base policy input = [proprio history (3*76), z_t history (3*8)] = 252 dim.
"""
import sys
import os
import time
import collections
import yaml
import torch
import numpy as np
import mujoco
import mujoco.viewer
# from legged_gym import LEGGED_GYM_ROOT_DIR

# Make `RMA.rma_modules` importable regardless of cwd. RMA lives at
# h12_adaptive_policy/RMA/, so we add the h12_adaptive_policy package dir
# (one level above deploy/) to sys.path.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)

RMA_LATENT_DIM = 8
RMA_ACTOR_Z_DIM = 24   # 3 * 8
RMA_ET_DIM = 21        # 15 upper dof + left_xyz(3) + right_xyz(3), hand-only
H12_POLICY_JOINTS = 27


def load_config(config_path):
    """Load and process the YAML configuration file (same as deploy_h12 + RMA keys)."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    for path_key in ["policy_path", "xml_path", "encoder_path"]:
        if path_key in config and config[path_key] and isinstance(config[path_key], str):
            config[path_key] = config[path_key]

    array_keys = ["kps", "kds", "default_angles", "cmd_scale", "cmd_init"]
    if "kps_arms" in config:
        array_keys.extend(["kps_arms", "kds_arms"])
    if "default_angles_arms" in config:
        array_keys.append("default_angles_arms")
    if "left_hand_force" in config:
        config["left_hand_force"] = np.array(config["left_hand_force"], dtype=np.float32)
    if "right_hand_force" in config:
        config["right_hand_force"] = np.array(config["right_hand_force"], dtype=np.float32)
    for key in array_keys:
        if key in config:
            config[key] = np.array(config[key], dtype=np.float32)
    return config


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def quat_rotate_inverse(q, v):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    q_conj = np.array([w, -x, -y, -z])
    return np.array([
        v[0] * (q_conj[0]**2 + q_conj[1]**2 - q_conj[2]**2 - q_conj[3]**2) +
        v[1] * 2 * (q_conj[1] * q_conj[2] - q_conj[0] * q_conj[3]) +
        v[2] * 2 * (q_conj[1] * q_conj[3] + q_conj[0] * q_conj[2]),
        v[0] * 2 * (q_conj[1] * q_conj[2] + q_conj[0] * q_conj[3]) +
        v[1] * (q_conj[0]**2 - q_conj[1]**2 + q_conj[2]**2 - q_conj[3]**2) +
        v[2] * 2 * (q_conj[2] * q_conj[3] - q_conj[0] * q_conj[1]),
        v[0] * 2 * (q_conj[1] * q_conj[3] - q_conj[0] * q_conj[2]) +
        v[1] * 2 * (q_conj[2] * q_conj[3] + q_conj[0] * q_conj[1]) +
        v[2] * (q_conj[0]**2 - q_conj[1]**2 - q_conj[2]**2 + q_conj[3]**2)
    ])


def get_gravity_orientation(quat):
    return quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))


def compute_observation(d, config, action, cmd, height_cmd, n_joints, qj=None, dqj=None):
    """Same as mujoco_deploy_h12: single obs 76 dim."""
    if qj is None:
        qj = d.qpos[7 : 7 + n_joints].copy()
    else:
        qj = qj.copy()
    if dqj is None:
        dqj = d.qvel[6 : 6 + n_joints].copy()
    else:
        dqj = dqj.copy()
    quat = d.qpos[3:7].copy()
    omega = d.qvel[3:6].copy()

    if len(config["default_angles"]) < n_joints:
        padded_defaults = np.zeros(n_joints, dtype=np.float32)
        padded_defaults[: len(config["default_angles"])] = config["default_angles"]
        if "default_angles_arms" in config and n_joints >= len(config["default_angles"]) + len(config["default_angles_arms"]):
            padded_defaults[len(config["default_angles"]) : len(config["default_angles"]) + len(config["default_angles_arms"])] = config["default_angles_arms"]
    else:
        padded_defaults = config["default_angles"][:n_joints]

    qj_scaled = (qj - padded_defaults) * config["dof_pos_scale"]
    dqj_scaled = dqj * config["dof_vel_scale"]
    gravity_orientation = get_gravity_orientation(quat)
    omega_scaled = omega * config["ang_vel_scale"]

    single_obs_dim = 3 + 1 + 3 + 3 + n_joints + n_joints + 12
    single_obs = np.zeros(single_obs_dim, dtype=np.float32)
    single_obs[0:3] = cmd[:3] * config["cmd_scale"]
    single_obs[3:4] = np.array([height_cmd])
    single_obs[4:7] = omega_scaled
    single_obs[7:10] = gravity_orientation
    single_obs[10 : 10 + n_joints] = qj_scaled
    single_obs[10 + n_joints : 10 + 2 * n_joints] = dqj_scaled
    single_obs[10 + 2 * n_joints : 10 + 2 * n_joints + 12] = action
    return single_obs, single_obs_dim


def build_et_mujoco(
    qpos,
    left_hand_force_xyz,
    right_hand_force_xyz,
    num_actions=12,
    policy_joints=H12_POLICY_JOINTS,
    qj_policy=None,
):
    """e_t = 15 upper-body dof + left_xyz(3) + right_xyz(3) = 21 (hand-only, same order as Isaac build_et_from_gym)."""
    if qj_policy is None:
        upper = qpos[7 + num_actions : 7 + policy_joints].copy()
    else:
        upper = qj_policy[num_actions:policy_joints].copy()
    return np.concatenate([upper, np.asarray(left_hand_force_xyz, dtype=np.float32), np.asarray(right_hand_force_xyz, dtype=np.float32)], dtype=np.float32)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.join(_SCRIPT_DIR, "h1_2_rma_arm.yaml"))
    parser.add_argument(
        "--no_encode",
        action="store_true",
        help="Feed encoder e_t with zero forces (forces still applied to sim). Test naive policy vs encoding.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    no_encode = args.no_encode or config.get("no_encode", False)
    if no_encode:
        print("no_encode=True: forces applied to robot, but e_t uses zeros for encoder (naive policy test).")

    # Resolve relative paths relative to config file directory
    config_dir = os.path.dirname(os.path.abspath(args.config))
    for key in ["policy_path", "xml_path", "encoder_path"]:
        if key in config and config[key] and isinstance(config[key], str) and not os.path.isabs(config[key]):
            config[key] = os.path.normpath(os.path.join(config_dir, config[key]))

    m = mujoco.MjModel.from_xml_path(config["xml_path"])
    d = mujoco.MjData(m)
    m.opt.timestep = config["simulation_dt"]

    n_joints_total = d.qpos.shape[0] - 7
    policy_joints = int(config.get("policy_num_joints", H12_POLICY_JOINTS))
    if n_joints_total < policy_joints:
        raise ValueError(f"Model has only {n_joints_total} joints, but policy expects {policy_joints} joints")

    h12_ctrl_count = int(config.get("h12_ctrl_count", policy_joints))
    if d.ctrl.shape[0] < h12_ctrl_count:
        raise ValueError(f"Model has only {d.ctrl.shape[0]} actuators, but h12_ctrl_count={h12_ctrl_count}")

    # Build stable index mapping for H12 joints from actuator transmissions.
    # This avoids assuming qpos/qvel are contiguous when extra joints (e.g., grippers)
    # are inserted into the kinematic tree.
    h12_joint_ids = m.actuator_trnid[:h12_ctrl_count, 0].astype(np.int32)
    h12_qpos_adr = m.jnt_qposadr[h12_joint_ids].astype(np.int32)
    h12_qvel_adr = m.jnt_dofadr[h12_joint_ids].astype(np.int32)

    leg_count = config["num_actions"]
    leg_qpos_adr = h12_qpos_adr[:leg_count]
    leg_qvel_adr = h12_qvel_adr[:leg_count]
    upper_qpos_adr = h12_qpos_adr[leg_count:h12_ctrl_count]
    upper_qvel_adr = h12_qvel_adr[leg_count:h12_ctrl_count]

    print(
        f"Model DOFs (qpos): {d.qpos.shape[0]}, joints(total): {n_joints_total}, "
        f"policy joints: {policy_joints}, ctrl size(total): {d.ctrl.shape[0]}, h12 ctrl count: {h12_ctrl_count}"
    )

    # RMA: body ids and forces (hand-only; no torso)
    left_wrist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_roll_link")
    right_wrist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")
    apply_forces = left_wrist_id >= 0 and right_wrist_id >= 0
    if not apply_forces:
        print("Warning: RMA force bodies (left/right_wrist_roll_link) not found; skipping xfrc_applied.")
    left_hand_force = config["left_hand_force"].copy()
    right_hand_force = config["right_hand_force"].copy()

    # RMA: load encoder
    encoder = None
    if config.get("encoder_path") and os.path.isfile(config["encoder_path"]):
        from RMA.rma_modules.env_factor_encoder import EnvFactorEncoder, EnvFactorEncoderCfg
        encoder = EnvFactorEncoder(EnvFactorEncoderCfg())
        encoder.load_state_dict(torch.load(config["encoder_path"], map_location="cpu", weights_only=True))
        encoder.eval()
        print(f"Loaded encoder from {config['encoder_path']}")
    else:
        print("No encoder_path or file not found; z_t will be zeros.")

    action = np.zeros(config["num_actions"], dtype=np.float32)
    target_dof_pos = config["default_angles"].copy()
    cmd = config["cmd_init"].copy()
    height_cmd = config["height_cmd"]

    qj_h12 = d.qpos[h12_qpos_adr]
    dqj_h12 = d.qvel[h12_qvel_adr]
    single_obs, single_obs_dim = compute_observation(
        d,
        config,
        action,
        cmd,
        height_cmd,
        policy_joints,
        qj=qj_h12,
        dqj=dqj_h12,
    )
    obs_history = collections.deque(maxlen=config["obs_history_len"])
    for _ in range(config["obs_history_len"]):
        obs_history.append(np.zeros(single_obs_dim, dtype=np.float32))

    # RMA: z history (3, 8)
    z_history = np.zeros((3, RMA_LATENT_DIM), dtype=np.float32)

    obs = np.zeros(config["num_obs"], dtype=np.float32)
    policy = torch.jit.load(config["policy_path"])
    print(policy)
    counter = 0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and (time.time() - start) < config["simulation_duration"]:
            step_start = time.time()

            # RMA: apply forces before step (hand-only; full 3D world frame: Fx, Fy, Fz)
            d.xfrc_applied[:] = 0
            if apply_forces:
                d.xfrc_applied[left_wrist_id, :3] = left_hand_force   # x, y, z all applied
                d.xfrc_applied[right_wrist_id, :3] = right_hand_force

            leg_tau = pd_control(
                target_dof_pos,
                d.qpos[leg_qpos_adr],
                config["kps"],
                np.zeros_like(config["kps"]),
                d.qvel[leg_qvel_adr],
                config["kds"],
            )
            leg_tau = np.nan_to_num(leg_tau, nan=0.0, posinf=0.0, neginf=0.0)
            max_tau = 200.0
            leg_tau = np.clip(leg_tau, -max_tau, max_tau)
            d.ctrl[: config["num_actions"]] = leg_tau

            upper_h12_count = h12_ctrl_count - config["num_actions"]
            if upper_h12_count > 0:
                kps_arm = config.get("kps_arms", np.ones(upper_h12_count, dtype=np.float32) * 500.0)
                kds_arm = config.get("kds_arms", np.ones(upper_h12_count, dtype=np.float32) * 5.0)
                arm_target_positions = config.get("default_angles_arms", np.zeros(upper_h12_count, dtype=np.float32))
                if len(arm_target_positions) < upper_h12_count:
                    arm_target_positions = np.zeros(upper_h12_count, dtype=np.float32)
                arm_tau = pd_control(
                    arm_target_positions[:upper_h12_count],
                    d.qpos[upper_qpos_adr],
                    kps_arm,
                    np.zeros(upper_h12_count),
                    d.qvel[upper_qvel_adr],
                    kds_arm,
                )
                arm_tau = np.nan_to_num(arm_tau, nan=0.0, posinf=0.0, neginf=0.0)
                arm_tau = np.clip(arm_tau, -max_tau, max_tau)
                d.ctrl[config["num_actions"] : h12_ctrl_count] = arm_tau

            # Keep any extra actuators (e.g., Magpie grippers) passive for 27-action policies.
            if d.ctrl.shape[0] > h12_ctrl_count:
                gripper_ctrlrange = m.actuator_ctrlrange[h12_ctrl_count:, :]
                d.ctrl[h12_ctrl_count:] = 0.5 * (gripper_ctrlrange[:, 0] + gripper_ctrlrange[:, 1])

            mujoco.mj_step(m, d)
            counter += 1

            if counter % config["control_decimation"] == 0:
                qj_h12 = d.qpos[h12_qpos_adr]
                dqj_h12 = d.qvel[h12_qvel_adr]
                single_obs, _ = compute_observation(
                    d,
                    config,
                    action,
                    cmd,
                    height_cmd,
                    policy_joints,
                    qj=qj_h12,
                    dqj=dqj_h12,
                )
                obs_history.append(single_obs)

                # RMA: e_t -> encoder -> z_t; update z history (hand-only 3D)
                if no_encode:
                    e_t = build_et_mujoco(
                        d.qpos,
                        np.zeros(3, dtype=np.float32),
                        np.zeros(3, dtype=np.float32),
                        config["num_actions"],
                        policy_joints,
                        qj_h12,
                    )
                else:
                    e_t = build_et_mujoco(
                        d.qpos,
                        left_hand_force,
                        right_hand_force,
                        config["num_actions"],
                        policy_joints,
                        qj_h12,
                    )
                if encoder is not None:
                    with torch.no_grad():
                        z_t = encoder(torch.from_numpy(e_t).unsqueeze(0).float()).numpy().squeeze()
                else:
                    z_t = np.zeros(RMA_LATENT_DIM, dtype=np.float32)
                z_history[1:, :] = z_history[:-1, :].copy()
                z_history[0, :] = z_t
                z_flat = np.flip(z_history, axis=0).flatten().astype(np.float32)  # [z_oldest, z_mid, z_new] -> 24

                # actor_obs = [proprio history 228, z_flat 24] = 252
                proprio = np.concatenate(list(obs_history), axis=0)
                actor_obs = np.concatenate([proprio, z_flat], axis=0).astype(np.float32)
                assert actor_obs.shape[0] == config["num_obs"], (actor_obs.shape[0], config["num_obs"])

                obs_tensor = torch.from_numpy(actor_obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze()

                if counter % (config["control_decimation"] * 50) == 0:
                    print(f"z_t: {z_t}")

                target_dof_pos = action * config["action_scale"] + config["default_angles"]

            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
