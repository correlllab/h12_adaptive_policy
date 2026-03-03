"""
6D random hand force sweep for RMA: sample (Fx,Fy,Fz) for left and right hand,
with Fx,Fy in [-xy_max, xy_max] and Fz in [-z_max, z_max], run sim, save CSV,
and plot 2D pairwise + 3D scatter.

Usage:
  # Default: Fx,Fy in [-10,10], Fz in [-30,30]
  python eval_rma_hand_sweep_6d.py --N 200 --duration 10

  # Custom ranges
  python eval_rma_hand_sweep_6d.py --N 200 --xy_max 10 --z_max 30 --duration 10
  python eval_rma_hand_sweep_6d.py --N 100 --csv results/hand_6d.csv --out results/hand_6d.png
"""

import sys
import os
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import mujoco
import torch
from mujoco_deploy_h12_rma import (
    load_config,
    pd_control,
    compute_observation,
    build_et_mujoco,
    get_gravity_orientation,
    RMA_LATENT_DIM,
)

HEIGHT_THRESHOLD = 0.55
TILT_DEG_THRESHOLD = 45.0


def get_tilt_deg(quat):
    gravity_body = get_gravity_orientation(quat)
    cos_tilt = np.clip(-gravity_body[2], -1.0, 1.0)
    return np.degrees(np.arccos(cos_tilt))


def run_one_vec(config, m, left_vec_3d, right_vec_3d, duration_s, policy, encoder,
                left_wrist_id, right_wrist_id, apply_forces, no_encode=False,
                n_joints=None, height_cmd_override=None):
    """Run sim for one 6D hand force (left_vec_3d, right_vec_3d) in world frame [Fx,Fy,Fz] N.
    Returns (success, time_to_fall_s).
    """
    d = mujoco.MjData(m)
    decim = config["control_decimation"]
    dt = config["simulation_dt"]
    n_joints = d.qpos.shape[0] - 7 if n_joints is None else n_joints
    left_force_3d = np.asarray(left_vec_3d, dtype=np.float32)
    right_force_3d = np.asarray(right_vec_3d, dtype=np.float32)

    action = np.zeros(config["num_actions"], dtype=np.float32)
    target_dof_pos = config["default_angles"].copy()
    cmd = config["cmd_init"].copy()
    height_cmd = float(height_cmd_override) if height_cmd_override is not None else config["height_cmd"]

    single_obs, single_obs_dim = compute_observation(d, config, action, cmd, height_cmd, n_joints)
    obs_history = __import__("collections").deque(maxlen=config["obs_history_len"])
    for _ in range(config["obs_history_len"]):
        obs_history.append(single_obs.copy())

    z_history = np.zeros((3, RMA_LATENT_DIM), dtype=np.float32)
    max_tau = 200.0
    n_steps = int(duration_s / dt)
    success = True
    time_to_fall_s = None

    for step in range(n_steps):
        d.xfrc_applied[:] = 0
        if apply_forces:
            d.xfrc_applied[left_wrist_id, :3] = left_force_3d
            d.xfrc_applied[right_wrist_id, :3] = right_force_3d

        leg_tau = pd_control(
            target_dof_pos,
            d.qpos[7 : 7 + config["num_actions"]],
            config["kps"],
            np.zeros_like(config["kps"]),
            d.qvel[6 : 6 + config["num_actions"]],
            config["kds"],
        )
        leg_tau = np.nan_to_num(leg_tau, nan=0.0, posinf=0.0, neginf=0.0)
        leg_tau = np.clip(leg_tau, -max_tau, max_tau)
        d.ctrl[: config["num_actions"]] = leg_tau

        if n_joints > config["num_actions"]:
            kps_arm = config.get("kps_arms", np.ones(n_joints - config["num_actions"], dtype=np.float32) * 500.0)
            kds_arm = config.get("kds_arms", np.ones(n_joints - config["num_actions"], dtype=np.float32) * 5.0)
            arm_target_positions = config.get("default_angles_arms", np.zeros(n_joints - config["num_actions"], dtype=np.float32))
            if len(arm_target_positions) < n_joints - config["num_actions"]:
                arm_target_positions = np.zeros(n_joints - config["num_actions"], dtype=np.float32)
            arm_tau = pd_control(
                arm_target_positions[: n_joints - config["num_actions"]],
                d.qpos[7 + config["num_actions"] : 7 + n_joints],
                kps_arm,
                np.zeros(n_joints - config["num_actions"]),
                d.qvel[6 + config["num_actions"] : 6 + n_joints],
                kds_arm,
            )
            arm_tau = np.nan_to_num(arm_tau, nan=0.0, posinf=0.0, neginf=0.0)
            arm_tau = np.clip(arm_tau, -max_tau, max_tau)
            d.ctrl[config["num_actions"] :] = arm_tau

        mujoco.mj_step(m, d)

        if step % decim == 0:
            single_obs, _ = compute_observation(d, config, action, cmd, height_cmd, n_joints)
            obs_history.append(single_obs)
            if no_encode:
                e_t = build_et_mujoco(d.qpos, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), config["num_actions"])
            else:
                e_t = build_et_mujoco(d.qpos, left_force_3d, right_force_3d, config["num_actions"])
            if encoder is not None:
                with torch.no_grad():
                    z_t = encoder(torch.from_numpy(e_t).unsqueeze(0).float()).numpy().squeeze()
            else:
                z_t = np.zeros(RMA_LATENT_DIM, dtype=np.float32)
            z_history[1:, :] = z_history[:-1, :].copy()
            z_history[0, :] = z_t
            z_flat = np.flip(z_history, axis=0).flatten().astype(np.float32)
            proprio = np.concatenate(list(obs_history), axis=0)
            actor_obs = np.concatenate([proprio, z_flat], axis=0).astype(np.float32)
            obs_tensor = torch.from_numpy(actor_obs).unsqueeze(0)
            action = policy(obs_tensor).detach().numpy().squeeze()
            target_dof_pos = action * config["action_scale"] + config["default_angles"]

        height = d.qpos[2]
        quat = d.qpos[3:7]
        tilt_deg = get_tilt_deg(quat)
        if height < HEIGHT_THRESHOLD or tilt_deg > TILT_DEG_THRESHOLD:
            success = False
            time_to_fall_s = step * dt
            break

    return success, time_to_fall_s


def main():
    parser = argparse.ArgumentParser(description="6D random hand force sweep: sample Fx,Fy,Fz both hands, plot 2D pairs + 3D scatter")
    parser.add_argument("--config", type=str, default=os.path.join(_SCRIPT_DIR, "h1_2_rma_arm.yaml"))
    parser.add_argument("--weights_dir", type=str, default=None, help="Directory with policy.pt and encoder_*.pt")
    parser.add_argument("--N", type=int, default=150, help="Number of random 6D force samples")
    parser.add_argument("--xy_max", type=float, default=20.0, help="Fx,Fy range: each in [-xy_max, xy_max] N")
    parser.add_argument("--z_max", type=float, default=30.0, help="Fz range: in [-z_max, z_max] N")
    parser.add_argument("--duration", type=float, default=10.0, help="Sim duration per condition (s)")
    parser.add_argument("--h_min", type=float, default=0.7, help="Min height_cmd for random sweep (m)")
    parser.add_argument("--h_max", type=float, default=1.0, help="Max height_cmd for random sweep (m)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    parser.add_argument("--no_encode", action="store_true", help="Encoder sees zero forces (naive policy)")
    parser.add_argument("--csv", type=str, default=None, help="Save results to this CSV (default: hand_sweep_6d.csv)")
    parser.add_argument("--out", type=str, default="hand_sweep_6d.png", help="Output plot path")
    parser.add_argument(
        "--arm_pose",
        type=str,
        choices=("forward_extended", "mid_sideways", "full_sideways", "asym_forward_full", "near_forward", "near_full_sideways"),
        default=None,
        help="Override arm default pose using arm_pose_presets in YAML",
    )
    args = parser.parse_args()

    os.chdir(_SCRIPT_DIR)
    config_path = args.config if os.path.isabs(args.config) else os.path.join(_SCRIPT_DIR, args.config)
    config = load_config(config_path)

    config_dir = os.path.dirname(os.path.abspath(config_path))
    for key in ["policy_path", "xml_path", "encoder_path"]:
        if key in config and config[key] and not os.path.isabs(config[key]):
            config[key] = os.path.normpath(os.path.join(config_dir, config[key]))

    # Optional arm pose override from YAML presets
    if args.arm_pose is not None:
        presets = config.get("arm_pose_presets", {})
        if args.arm_pose in presets:
            import numpy as _np
            config["default_angles_arms"] = _np.array(presets[args.arm_pose], dtype=_np.float32)

    if args.weights_dir:
        wd = args.weights_dir if os.path.isabs(args.weights_dir) else os.path.join(_SCRIPT_DIR, args.weights_dir)
        policy_candidate = os.path.join(wd, "policy.pt")
        if os.path.isfile(policy_candidate):
            config["policy_path"] = policy_candidate
            print(f"Using policy: {config['policy_path']}")
        encoder_files = sorted(glob.glob(os.path.join(wd, "encoder_*.pt")))
        if encoder_files:
            preferred = [p for p in encoder_files if "encoder_4999" in p]
            config["encoder_path"] = preferred[0] if preferred else encoder_files[-1]
            print(f"Using encoder: {config['encoder_path']}")

    m = mujoco.MjModel.from_xml_path(config["xml_path"])
    m.opt.timestep = config["simulation_dt"]
    n_joints = m.nq - 7

    left_wrist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_roll_link")
    right_wrist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_roll_link")
    apply_forces = left_wrist_id >= 0 and right_wrist_id >= 0

    policy = torch.jit.load(config["policy_path"])
    encoder = None
    if config.get("encoder_path") and os.path.isfile(config["encoder_path"]):
        from RMA.rma_modules.env_factor_encoder import EnvFactorEncoder, EnvFactorEncoderCfg
        encoder = EnvFactorEncoder(EnvFactorEncoderCfg())
        encoder.load_state_dict(torch.load(config["encoder_path"], map_location="cpu", weights_only=True))
        encoder.eval()

    no_encode_flag = args.no_encode
    label = "naive" if no_encode_flag else "RMA"
    mode_tag = "naive" if no_encode_flag else "rma"
    xy_max = args.xy_max
    z_max = args.z_max
    h_min = args.h_min
    h_max = args.h_max

    # Sample 6D: left (Fx,Fy,Fz), right (Fx,Fy,Fz)
    # Fx,Fy in [-xy_max, xy_max], Fz in [-z_max, z_max]
    np.random.seed(args.seed)
    left_fx = np.random.uniform(-xy_max, xy_max, size=(args.N,))
    left_fy = np.random.uniform(-xy_max, xy_max, size=(args.N,))
    left_fz = np.random.uniform(-z_max, z_max, size=(args.N,))
    right_fx = np.random.uniform(-xy_max, xy_max, size=(args.N,))
    right_fy = np.random.uniform(-xy_max, xy_max, size=(args.N,))
    right_fz = np.random.uniform(-z_max, z_max, size=(args.N,))
    heights = np.random.uniform(h_min, h_max, size=(args.N,))
    left_forces = np.stack([left_fx, left_fy, left_fz], axis=1)
    right_forces = np.stack([right_fx, right_fy, right_fz], axis=1)

    results = {
        "Fx_L": left_forces[:, 0],
        "Fy_L": left_forces[:, 1],
        "Fz_L": left_forces[:, 2],
        "Fx_R": right_forces[:, 0],
        "Fy_R": right_forces[:, 1],
        "Fz_R": right_forces[:, 2],
        "height_cmd": heights,
        "success": [],
        "time_to_fall": [],
    }

    for idx in range(args.N):
        ok, t_fall = run_one_vec(
            config, m,
            left_forces[idx], right_forces[idx],
            args.duration,
            policy, encoder,
            left_wrist_id, right_wrist_id, apply_forces,
            no_encode=no_encode_flag, n_joints=n_joints,
            height_cmd_override=heights[idx],
        )
        results["success"].append(1 if ok else 0)
        results["time_to_fall"].append(t_fall if t_fall is not None else args.duration)
        if (idx + 1) % 20 == 0 or idx == 0 or idx == args.N - 1:
            print(f"[{label}] {idx+1}/{args.N} L=({left_forces[idx,0]:.1f},{left_forces[idx,1]:.1f},{left_forces[idx,2]:.1f}) "
                  f"R=({right_forces[idx,0]:.1f},{right_forces[idx,1]:.1f},{right_forces[idx,2]:.1f}) -> "
                  f"{'OK' if ok else 'FALL'}" + (f" t={t_fall:.1f}s" if t_fall else ""))

    results["success"] = np.array(results["success"])
    results["time_to_fall"] = np.array(results["time_to_fall"])

    # Save CSV
    csv_path = args.csv
    if csv_path is None:
        arm_pose_tag = args.arm_pose if args.arm_pose is not None else "defaultpose"
        default_csv_name = f"hand_sweep_6d_{mode_tag}_{arm_pose_tag}.csv"
        csv_path = os.path.join(_SCRIPT_DIR, default_csv_name)
    elif not os.path.isabs(csv_path):
        csv_path = os.path.join(_SCRIPT_DIR, csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("Fx_L,Fy_L,Fz_L,Fx_R,Fy_R,Fz_R,height_cmd,success,time_to_fall\n")
        for i in range(args.N):
            f.write(f"{results['Fx_L'][i]},{results['Fy_L'][i]},{results['Fz_L'][i]},"
                    f"{results['Fx_R'][i]},{results['Fy_R'][i]},{results['Fz_R'][i]},"
                    f"{results['height_cmd'][i]},{results['success'][i]},{results['time_to_fall'][i]}\n")
    print(f"Saved CSV to {csv_path}")

    ok = results["success"] == 1
    t_fall = results["time_to_fall"]

    # Print aggregate success stats
    success_rate = float(results["success"].mean()) * 100.0
    num_success = int(results["success"].sum())
    print(f"Success rate: {success_rate:.1f}% ({num_success}/{args.N})")

    # Base path for 3D figure
    if args.out == "hand_sweep_6d.png":
        arm_pose_tag = args.arm_pose if args.arm_pose is not None else "defaultpose"
        out_basename = f"hand_sweep_6d_{mode_tag}_{arm_pose_tag}.png"
    else:
        out_basename = args.out
    out_path1 = out_basename if os.path.isabs(out_basename) else os.path.join(_SCRIPT_DIR, out_basename)

    # ---- Figure 2: two 3D scatter plots (left hand forces, right hand forces), colored by success
    fig2 = plt.figure(figsize=(12, 5))
    ax_left = fig2.add_subplot(121, projection="3d")
    ax_right = fig2.add_subplot(122, projection="3d")

    ax_left.scatter(results["Fx_L"][ok], results["Fy_L"][ok], results["Fz_L"][ok], c="green", s=20, alpha=0.7, label="Stand")
    ax_left.scatter(results["Fx_L"][~ok], results["Fy_L"][~ok], results["Fz_L"][~ok], c="red", s=20, alpha=0.7, label="Fall")
    ax_left.set_xlabel("Fx_L (N)")
    ax_left.set_ylabel("Fy_L (N)")
    ax_left.set_zlabel("Fz_L (N)")
    ax_left.set_title("Left hand (Fx, Fy, Fz)")
    ax_left.legend(loc="upper right", fontsize=8)

    ax_right.scatter(results["Fx_R"][ok], results["Fy_R"][ok], results["Fz_R"][ok], c="green", s=20, alpha=0.7, label="Stand")
    ax_right.scatter(results["Fx_R"][~ok], results["Fy_R"][~ok], results["Fz_R"][~ok], c="red", s=20, alpha=0.7, label="Fall")
    ax_right.set_xlabel("Fx_R (N)")
    ax_right.set_ylabel("Fy_R (N)")
    ax_right.set_zlabel("Fz_R (N)")
    ax_right.set_title("Right hand (Fx, Fy, Fz)")
    ax_right.legend(loc="upper right", fontsize=8)

    fig2.suptitle(f"6D hand force sweep — 3D view, N={args.N}" + (" [no_encode]" if no_encode_flag else " [RMA]"))
    plt.tight_layout()
    base, ext = os.path.splitext(out_path1)
    out_path2 = base + "_3d" + ext
    fig2.savefig(out_path2, dpi=150)
    print(f"Saved 3D plot to {out_path2}")

    plt.show()


if __name__ == "__main__":
    main()
