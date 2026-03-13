# from legged_gym import LEGGED_GYM_ROOT_DIR
"""
Real-robot deploy: squat policy only.
Loads config (e.g. h1_2_real.yaml) and runs the squat policy on the H12 via DDS.
"""

import os
import sys
import argparse
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from typing import Union
import numpy as np
import time
import torch

import collections


from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from common.keyboard_controller import KeyboardRemoteController, print_keyboard_mapping
from config import Config

# RMA imports (encoder) – same as in MujocoDeploy
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# Add submodule to path for RobotModel
sys.path.append(os.path.join(_REPO_ROOT, 'submodules/h12_ros2_controller'))

from h12_adaptive_policy.RMA.rma_modules.env_factor_encoder import EnvFactorEncoder, EnvFactorEncoderCfg
from h12_ros2_controller.core.robot_model import RobotModel

RMA_LATENT_DIM = 8
RMA_ACTOR_Z_DIM = 24  # 3 * 8

######################################################################
## Plotting Configuration
######################################################################

# Joint names for plotting labels
JOINT_NAMES_PLOT = [
    "L_hip_yaw", "L_hip_pitch", "L_hip_roll", "L_knee", "L_ankle_pitch", "L_ankle_roll",
    "R_hip_yaw", "R_hip_pitch", "R_hip_roll", "R_knee", "R_ankle_pitch", "R_ankle_roll"
]

KNEE_ANKLE_LEG_IDXS = [3, 4, 5, 9, 10, 11]
KNEE_ANKLE_JOINT_NAMES = [JOINT_NAMES_PLOT[i] for i in KNEE_ANKLE_LEG_IDXS]

######################################################################
## Utility Functions
######################################################################

def plot_qpos_vs_action(t, qpos_hist, target_dof_hist, joint_names, save_path):
    """Plots measured joint positions against commanded positions."""
    n_joints = len(joint_names)
    fig, axes = plt.subplots(n_joints, 1, figsize=(12, 2.5 * n_joints), sharex=True)
    if n_joints == 1: axes = [axes]
    for i, name in enumerate(joint_names):
        axes[i].plot(t, qpos_hist[:, i], label="qpos (measured)", color="blue")
        axes[i].plot(t, target_dof_hist[:, i], label="target_dof_pos (command)", color="green", linestyle="--")
        axes[i].set_ylabel(name)
        axes[i].grid(True)
        if i == 0: axes[i].set_title("Measured Joint Position vs. Commanded Action")
        axes[i].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Saved overlay plot to {save_path}")
    plt.close(fig)

def plot_dqpos(t, dqpos_hist, joint_names, save_path):
    """Plots measured joint velocities."""
    # (Implementation is identical to your original code)
    n_joints = len(joint_names)
    fig, axes = plt.subplots(n_joints, 1, figsize=(12, 2.5 * n_joints), sharex=True)
    if n_joints == 1: axes = [axes]
    for i, name in enumerate(joint_names):
        axes[i].plot(t, dqpos_hist[:, i], label="dqpos (measured)", color="orange")
        axes[i].set_ylabel(name)
        axes[i].grid(True)
        if i == 0: axes[i].set_title("Measured Joint Velocity")
        axes[i].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Saved velocity plot to {save_path}")
    plt.close(fig)

######################################################################
## Main Controller Class
######################################################################
class Controller:
    def __init__(self, config: Config, use_keyboard: bool = False) -> None:
        self.config = config
        self.use_keyboard = use_keyboard
        if self.use_keyboard:
            self.remote_controller = KeyboardRemoteController()
        else:
            self.remote_controller = RemoteController()

        # Initialize the policy network
        self.policy = torch.jit.load(config.policy_path)

        # Optional RMA encoder (if encoder_path provided)
        self.encoder = None
        self.use_rma = config.encoder_path is not None and config.num_obs > config.single_obs_dim * config.obs_history_len
        if self.use_rma:
            self.encoder = EnvFactorEncoder(EnvFactorEncoderCfg())
            self.encoder.load_state_dict(torch.load(config.encoder_path, map_location="cpu", weights_only=True))
            self.encoder.eval()

        # Initializing process variables
        self.qj = np.zeros(config.num_dofs, dtype=np.float32)
        self.dqj = np.zeros(config.num_dofs, dtype=np.float32)
        self.tauj = np.zeros(config.num_dofs, dtype=np.float32)

        self.action = np.zeros(config.num_actions, dtype=np.float32)

        # RL observation vector (full size with history)
        self.target_dof_pos = config.default_angles.copy()

        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = config.cmd_init.copy()
        self.height_cmd = config.height_cmd  # constant from training (1.0)
        self.counter = 0

        # Histories for data logging
        self.qpos_hist, self.dq_hist, self.target_dof_hist, self.t_hist = [], [], [], []
        self.tau_hist = []
        self.knee_ankle_tau_hist = []
        self.start_time = time.time()

        self.single_obs_dim = config.single_obs_dim
        self.obs_history = collections.deque(maxlen=config.obs_history_len)

        # RMA latent history (z_{t-2}, z_{t-1}, z_t) if used
        if self.use_rma:
            self.z_history = np.zeros((3, RMA_LATENT_DIM), dtype=np.float32)

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # start robot model
        self.robot_model = RobotModel('./submodules/h12_ros2_controller/assets/h1_2/h1_2_handless.urdf')
        self.robot_model.init_subscriber()

        # wait for the subscriber to receive data
        print("waiting")
        self.wait_for_low_state()
        print("wait complete")
        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        if not self.use_keyboard:
            self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        if not self.use_keyboard:
            self.remote_controller.set(self.low_state.wireless_remote)

    def close(self):
        if hasattr(self.remote_controller, "close"):
            self.remote_controller.close()

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("▶️ Entering zero torque state. Press 'START' on controller to proceed...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("▶️ Moving to default position...")
        total_time = 2
        num_step = int(total_time / self.config.control_dt)

        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)
        dof_size = len(dof_idx)

        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q

        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
        print("✅ Reached default position.")

    def default_pos_state(self):
        print("▶️ Holding default position. Press 'A' on controller to start RL policy...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
        print("✅ RL Policy Engaged!")

    def run(self):
        self.counter += 1
        t_start = time.time()

        full_default_angles = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)

        # Get the current joint position and velocity
        all_motor_indices = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx

        for i, motor_idx in enumerate(all_motor_indices):
            # This populates self.qj[0:20] and self.dqj[0:20] (assuming 20 total DOFs)
            self.qj[i] = self.low_state.motor_state[motor_idx].q
            self.dqj[i] = self.low_state.motor_state[motor_idx].dq
            self.tauj[i] = self.low_state.motor_state[motor_idx].tau_est

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        if self.config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)

        # create observation (layout matches Isaac / MuJoCo)
        gravity_orientation = get_gravity_orientation(quat)

        num_dofs = self.config.num_dofs   # 27 (qj/dqj size)
        num_actions = self.config.num_actions # 12 (Policy action size, usually legs)

        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()

        qj_obs = (qj_obs - full_default_angles) * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        ang_vel_scaled = ang_vel * self.config.ang_vel_scale

        cmd_scaled = self.cmd * self.config.cmd_scale  # same as h12_controller_squat

        # Single observation size = 3 + 1 + 3 + 3 + 27 + 27 + 12 = 76 (assuming 12 actions)
        single_obs_dim = self.single_obs_dim
        single_obs = np.zeros(single_obs_dim, dtype=np.float32)

        current_idx = 0
        single_obs[current_idx : current_idx + 3] = cmd_scaled
        current_idx += 3

        single_obs[current_idx] = self.height_cmd
        current_idx += 1

        # 2. omega_scaled (3 values)
        single_obs[current_idx : current_idx + 3] = ang_vel_scaled
        current_idx += 3

        # 3. grav_orientation (3 values)
        single_obs[current_idx : current_idx + 3] = gravity_orientation
        current_idx += 3

        # 4. qj_scaled (27 values)
        single_obs[current_idx : current_idx + num_dofs] = qj_obs
        current_idx += num_dofs

        # 5. dqj_scaled (27 values)
        single_obs[current_idx : current_idx + num_dofs] = dqj_obs
        current_idx += num_dofs


         # 6. last_action (12 values)
        single_obs[current_idx : current_idx + num_actions] = self.action
        # current_idx += num_actions # Not strictly needed if this is the end

        # --- 4. HISTORY & STACK ---
        self.obs_history.append(single_obs.copy())

        # Construct full observation with history (proprio stack)
        proprio = np.zeros(self.config.num_obs, dtype=np.float32)
        for i, hist_obs in enumerate(self.obs_history):
            start_idx = i * single_obs_dim
            proprio[start_idx : start_idx + single_obs_dim] = hist_obs

        if self.use_rma and self.encoder is not None:
            # Build e_t for real robot: upper-body joint pos (15) + hand forces (left/right, 3D each) = 21
            num_actions = self.config.num_actions  # 12
            # upper-body joints: joints after 12 legs in qj (27 total)
            upper_pos = self.qj[num_actions : num_actions + 15].copy()

            # get hand forces from robot model
            # self.robot_model.update_kinematics()
            # left_force = -self.robot_model.get_frame_wrench("left_wrist_yaw_link")[0:3]
            # right_force = -self.robot_model.get_frame_wrench("right_wrist_yaw_link")[0:3]
            # print(f"Left hand force: {left_force}, Right hand force: {right_force}")
            # left_force = np.array([17.48624269, 7.02638068, -59.19414214])
            # right_force = np.array([-9.99245771, -4.1227193, -2.88810539])
            # left_force = np.array([-115.08508926, -30.89641282, 532.00016259])
            # right_force = np.array([74.82097678, 34.1630265, -37.48545911])

            # # random hand forces for testing
            # left_force = np.random.uniform(-500, 500, size=3)
            # right_force = np.random.uniform(-500, 500, size=3)

            # # hardcode hand forces
            # left_force = self.config.left_hand_force.astype(np.float32)
            # right_force = self.config.right_hand_force.astype(np.float32)

            # zero hand forces
            left_force = np.zeros(3, dtype=np.float32)
            right_force = np.zeros(3, dtype=np.float32)

            e_t = np.concatenate([upper_pos, left_force, right_force], dtype=np.float32)
            if self.config.no_encode:
                e_t[:] = 0.0

            print(e_t)
            with torch.no_grad():
                z_t = self.encoder(torch.from_numpy(e_t).unsqueeze(0).float()).numpy().squeeze()
            self.z_history[1:, :] = self.z_history[:-1, :].copy()
            self.z_history[0, :] = z_t
            z_flat = np.flip(self.z_history, axis=0).flatten().astype(np.float32)

            # actor_obs = np.concatenate([proprio, z_flat], axis=0).astype(np.float32)
            actor_obs = proprio.copy()
            actor_obs[single_obs_dim*3: single_obs_dim * 3 + len(z_flat)] = z_flat
        else:
            actor_obs = proprio.astype(np.float32)

        # --- 5. POLICY INFERENCE ---
        obs_tensor = torch.from_numpy(actor_obs).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().cpu().numpy().squeeze()

        scaled_action = self.action * self.config.action_scale

        clipped_action = np.clip(
                            scaled_action,
                            np.array(self.config.legs_motor_pos_lower_limit_list),
                            np.array(self.config.legs_motor_pos_upper_limit_list))

        target_dof_pos = self.config.default_angles + clipped_action



        # --- DATA LOGGING ADDITIONS ---
        current_time = time.time() - self.start_time
        self.t_hist.append(current_time)
        self.qpos_hist.append(self.qj.copy()) # Full qpos (27 DOFs)
        self.dq_hist.append(self.dqj.copy()) # Full dq (27 DOFs)
        self.tau_hist.append(self.tauj.copy()) # Full tau (27 DOFs)
        self.knee_ankle_tau_hist.append(self.tauj[KNEE_ANKLE_LEG_IDXS].copy())
        full_target_dof = np.concatenate((target_dof_pos, self.config.arm_waist_target), axis=0)
        self.target_dof_hist.append(full_target_dof)
        # -----------------------------


        # if self.counter <= 5:
        #     print(f"[{self.counter}] target dof: {target_dof_pos}")

        # Build low cmd
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        if self.counter <= 2:
            print(f"[{self.counter}] obs_history: {self.obs_history}")

        self.send_cmd(self.low_cmd)

        t_elapsed = time.time() - t_start
        time_to_sleep = self.config.control_dt - t_elapsed
        if time_to_sleep > 0:
             time.sleep(time_to_sleep)

        # Now, measure the time of the ENTIRE cycle, including the sleep
        t_full_cycle = time.time() - t_start

        # Print the full cycle time and the corresponding fixed frequency (50 Hz)
        print(f"Full Cycle Time: {t_full_cycle:.5f}s (Target: {self.config.control_dt:.3f}s), Frequency: {1.0/t_full_cycle:.1f} Hz")


if __name__ == "__main__":
    _SCRIPT_DIR = Path(__file__).resolve().parent
    _ROOT_DIR = _SCRIPT_DIR.parent
    parser = argparse.ArgumentParser(description="Run squat policy on H12 real robot.")
    parser.add_argument("net", type=str, help="Network interface for DDS")
    parser.add_argument("config", type=str, nargs="?", default="RealDeploy/h1_2_real.yaml",
                        help="Config YAML path (root-relative or RealDeploy-relative)")
    parser.add_argument("--save", type=str, required=True,
                        help="Log subfolder name under data/real (e.g. run_001)")
    parser.add_argument("--keyboard", action="store_true",
                        help="Use keyboard listener instead of wireless remote")
    args = parser.parse_args()

    folder_name = args.save.strip()
    folder_path = Path(folder_name)
    if (
        not folder_name
        or folder_path.is_absolute()
        or len(folder_path.parts) != 1
        or folder_name in {".", ".."}
    ):
        sys.exit("Invalid --save. Please provide a single folder name (no path separators).")

    config_arg = Path(args.config)
    candidate_paths = []
    if config_arg.is_absolute():
        candidate_paths.append(config_arg)
    else:
        candidate_paths.append(_ROOT_DIR / config_arg)
        candidate_paths.append(_SCRIPT_DIR / config_arg)

    config_path = next((path for path in candidate_paths if path.is_file()), None)
    if config_path is None:
        checked = "\n".join(str(path) for path in candidate_paths)
        sys.exit(f"Config not found. Checked:\n{checked}")
    config = Config(str(config_path))

    print("Config:", config.action_scale, config.cmd_scale, config.dof_pos_scale, config.dof_vel_scale, config.ang_vel_scale)
    print("Policy:", config.policy_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    if args.keyboard:
        print("Keyboard remote simulation enabled.")
        print_keyboard_mapping()

    controller = Controller(config, use_keyboard=args.keyboard)

    if args.keyboard and hasattr(controller.remote_controller, "_backend"):
        print(f"Keyboard backend: {controller.remote_controller._backend}")

    controller.zero_torque_state()

    controller.move_to_default_pos()

    controller.default_pos_state()

    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    controller.close()
    print("Exit")


    # ----------------------------------------------------------------------------------
    # --- POST-EXECUTION LOG SAVE ---
    # ----------------------------------------------------------------------------------
    log_dir = os.path.join("data", "real", folder_name)
    os.makedirs(log_dir, exist_ok=True)

    qpos_hist_arr = np.array(controller.qpos_hist)
    dq_hist_arr = np.array(controller.dq_hist)
    tau_hist_arr = np.array(controller.tau_hist)
    target_dof_hist_arr = np.array(controller.target_dof_hist)
    knee_ankle_tau_hist_arr = np.array(controller.knee_ankle_tau_hist)
    t_hist_arr = np.array(controller.t_hist)

    if not t_hist_arr.size:
        print("No data logged, skipping save.")
        sys.exit() # Exit after cleanup and message if no data

    qpos_save_path = os.path.join(log_dir, "qpos.npy")
    dq_save_path = os.path.join(log_dir, "dq.npy")
    tau_save_path = os.path.join(log_dir, "tau.npy")
    target_dof_save_path = os.path.join(log_dir, "target_dof.npy")
    time_save_path = os.path.join(log_dir, "time.npy")

    np.save(qpos_save_path, qpos_hist_arr)
    np.save(dq_save_path, dq_hist_arr)
    np.save(tau_save_path, tau_hist_arr)
    np.save(target_dof_save_path, target_dof_hist_arr)
    np.save(time_save_path, t_hist_arr)

    print(f"✅ Saved qpos log to {qpos_save_path}")
    print(f"✅ Saved dq log to {dq_save_path}")
    print(f"✅ Saved tau log to {tau_save_path}")
    print(f"✅ Saved target_dof log to {target_dof_save_path}")
    print(f"✅ Saved time log to {time_save_path}")

    # Plotting disabled by request.
    # plot_qpos_vs_action(
    #     t_hist_arr,
    #     qpos_hist_arr,
    #     target_dof_hist_arr,
    #     JOINT_NAMES_PLOT,
    #     os.path.join(log_dir, "qpos_vs_target.png")
    # )

    # plot_dqpos(
    #     t_hist_arr,
    #     dq_hist_arr,
    #     JOINT_NAMES_PLOT,
    #     os.path.join(log_dir, "dqpos.png")
    # )

    if knee_ankle_tau_hist_arr.size:
        knee_tau_save_path = os.path.join(log_dir, "knee_ankle_tau.csv")
        tau_log = np.column_stack((t_hist_arr, knee_ankle_tau_hist_arr))
        tau_header = "time," + ",".join(KNEE_ANKLE_JOINT_NAMES)
        np.savetxt(knee_tau_save_path, tau_log, delimiter=",", header=tau_header, comments="")
        print(f"✅ Saved knee/ankle tau log to {knee_tau_save_path}")
