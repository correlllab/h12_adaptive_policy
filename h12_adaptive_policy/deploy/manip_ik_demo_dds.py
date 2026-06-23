"""
DDS-decoupled single-arm pick-place + FAME controller.

Mirror of `manip_ik_demo.py`, but communicates with the simulation through DDS
topics instead of poking MuJoCo state directly. Same control code can target
`h1_mujoco` (sim) or a real Unitree H1_2 by swapping the network interface.

Inputs from sim:
  rt/lowstate         (LowState_)        — 27 motor q/dq/tau_est + IMU
  rt/sportmodestate   (SportModeState_)  — IMU world position
Outputs to sim:
  rt/lowcmd           (LowCmd_)          — 27 motor PD commands

Sim assumption: `h1_mujoco/h12_mujoco.py` runs in a separate process with its
default scene (`scene_handless.xml` or `scene_handless_pelvis_fixed.xml`). No
Magpie gripper in the sim — the controller still uses the handless URDF for IK,
which matches what `manip_ik_demo.py` already does for the IK chain.
Watchdog: must publish a LowCmd at least every 100 ms.

Limitation vs `manip_ik_demo.py`: the synthetic payload force (`d.xfrc_applied`)
is NOT applied to the sim — h1_mujoco has no DDS topic for force injection. The
encoder still receives the computed F = kg*g via e_t (so FAME's latent adapts),
but sim physics is unloaded. Documented; can be restored by adding a topic to
`h1_mujoco/unitree_interface.py` later.

Usage:
  # Terminal A: sim (pick --fixed for the first smoke test — pelvis welded)
  cd ~/isaac_gym_projects/h1_mujoco
  python h12_mujoco.py --fixed
  # Terminal B: controller
  python h12_adaptive_policy/deploy/manip_ik_demo_dds.py --task right_hand_manip
"""
import sys
import os
import argparse
import threading
import time
import yaml
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
for p in (_SCRIPT_DIR, os.path.join(_REPO_ROOT, "h12_adaptive_policy"),
          os.path.join(_REPO_ROOT, "submodules", "h12_ros2_controller")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from scipy.spatial.transform import Rotation as R

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize, ChannelSubscriber, ChannelPublisher,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.utils.crc import CRC

from mujoco_deploy_h12_rma import (
    load_config, clip_policy_action, compute_observation, build_et_mujoco,
    get_gravity_orientation, RMA_LATENT_DIM,
)
from RMA.rma_modules.env_factor_encoder import EnvFactorEncoder, EnvFactorEncoderCfg
from h12_ros2_controller.core.robot_model import RobotModel
from utils import (
    quat2R, smoothstep,
    ArmIK, solve_arm_waypoints, build_xyz_schedule, build_payload_profile, ARM_SLICE,
    plot_traj, plot_summary, plot_adapt, plot_adapt_single,
)

# YAML fallback topic names. The active values come from the deploy config.
DEFAULT_LOW_CMD_TOPIC = "rt/lowcmd"
DEFAULT_LOW_STATE_TOPIC = "rt/lowstate"
DEFAULT_HIGH_STATE_TOPIC = "rt/sportmodestate"

# IMU site offset in torso_link frame (from h1_2_magpie_fame.xml / h1_2.xml).
P_IMU_IN_TORSO = np.array([-0.04452, -0.01891, 0.27756])

# h1_mujoco hard-codes the sim timestep to 5 ms (mujoco_env.py:23).
SIM_DT = 0.005
# Policy step at 50 Hz on a 200 Hz sim. (manip_ik_demo.py YAML had decim=10 at 500 Hz sim.)
POLICY_DECIM = 4

HEIGHT_THRESHOLD = 0.55
TILT_DEG_THRESHOLD = 45.0

N_MOTORS = 27
TORSO_MOTOR_IDX = 12  # 13th motor in the unitree HG order

# ─── Force source for the encoder ───────────────────────────────────────────
# Default = PRIVILEGED: the encoder is fed F = (payload_kg from yaml) × g, derived
# from the task spec — exactly what the FAME training-time encoder saw. Good for
# sim ablations.
#
# YAML fallback for encoder force source. The active value comes from the deploy config.
DEFAULT_FORCE_ESTIMATOR_ENABLED = False


# ─── DDS state snapshot ─────────────────────────────────────────────────────

class DDSState:
    """Lock-protected snapshot of the latest LowState + SportModeState."""

    def __init__(self):
        self.lock = threading.RLock()
        self.q = np.zeros(N_MOTORS, dtype=np.float32)
        self.dq = np.zeros(N_MOTORS, dtype=np.float32)
        self.tau_est = np.zeros(N_MOTORS, dtype=np.float32)
        self.imu_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # wxyz
        self.imu_gyro = np.zeros(3, dtype=np.float32)
        self.imu_acc = np.zeros(3, dtype=np.float32)
        self.imu_pos = np.zeros(3, dtype=np.float32)
        self.tick = 0
        self.have_low_state = False
        self.have_high_state = False
        self.new_state_event = threading.Event()

    def snapshot(self):
        with self.lock:
            return {
                "q": self.q.copy(),
                "dq": self.dq.copy(),
                "tau_est": self.tau_est.copy(),
                "imu_quat": self.imu_quat.copy(),
                "imu_gyro": self.imu_gyro.copy(),
                "imu_acc": self.imu_acc.copy(),
                "imu_pos": self.imu_pos.copy(),
                "tick": self.tick,
            }


def make_low_state_callback(state: DDSState):
    def cb(msg: LowStateHG):
        with state.lock:
            for i in range(N_MOTORS):
                state.q[i] = msg.motor_state[i].q
                state.dq[i] = msg.motor_state[i].dq
                state.tau_est[i] = msg.motor_state[i].tau_est
            state.imu_quat[0] = msg.imu_state.quaternion[0]
            state.imu_quat[1] = msg.imu_state.quaternion[1]
            state.imu_quat[2] = msg.imu_state.quaternion[2]
            state.imu_quat[3] = msg.imu_state.quaternion[3]
            state.imu_gyro[0] = msg.imu_state.gyroscope[0]
            state.imu_gyro[1] = msg.imu_state.gyroscope[1]
            state.imu_gyro[2] = msg.imu_state.gyroscope[2]
            state.imu_acc[0] = msg.imu_state.accelerometer[0]
            state.imu_acc[1] = msg.imu_state.accelerometer[1]
            state.imu_acc[2] = msg.imu_state.accelerometer[2]
            state.tick = int(msg.tick)
            state.have_low_state = True
        state.new_state_event.set()
    return cb


def make_high_state_callback(state: DDSState):
    def cb(msg: SportModeState_):
        with state.lock:
            state.imu_pos[0] = msg.position[0]
            state.imu_pos[1] = msg.position[1]
            state.imu_pos[2] = msg.position[2]
            state.have_high_state = True
    return cb


# ─── Pose recovery ──────────────────────────────────────────────────────────

def pelvis_from_imu(imu_pos_W, imu_quat_W_wxyz, q_torso):
    """Recover pelvis pose in world from the IMU pose + torso-joint angle.

    The IMU sits at p_imu_in_torso relative to torso_link; torso_link's origin
    coincides with pelvis's; torso_joint is a pure z-yaw between them. So
        R_pelvis = R_imu · Rz(-q_torso)
        p_pelvis = p_imu  - R_imu · p_imu_in_torso
    """
    imu_quat_xyzw = [imu_quat_W_wxyz[1], imu_quat_W_wxyz[2],
                     imu_quat_W_wxyz[3], imu_quat_W_wxyz[0]]
    R_imu = R.from_quat(imu_quat_xyzw).as_matrix()
    R_pelvis = R_imu @ R.from_euler("z", -float(q_torso)).as_matrix()
    p_pelvis = np.asarray(imu_pos_W, dtype=np.float64) - R_imu @ P_IMU_IN_TORSO
    q_pelvis_xyzw = R.from_matrix(R_pelvis).as_quat()
    q_pelvis_wxyz = np.array([q_pelvis_xyzw[3], q_pelvis_xyzw[0],
                              q_pelvis_xyzw[1], q_pelvis_xyzw[2]], dtype=np.float64)
    return p_pelvis, q_pelvis_wxyz


# ─── Command publish ────────────────────────────────────────────────────────

def fill_low_cmd(cmd_msg, q_des, dq_des, kp, kd, tau):
    """Populate cmd_msg in-place. All inputs are 27-vectors (or broadcastable scalars)."""
    q_des = np.broadcast_to(q_des, (N_MOTORS,)).astype(np.float32)
    dq_des = np.broadcast_to(dq_des, (N_MOTORS,)).astype(np.float32)
    kp = np.broadcast_to(kp, (N_MOTORS,)).astype(np.float32)
    kd = np.broadcast_to(kd, (N_MOTORS,)).astype(np.float32)
    tau = np.broadcast_to(tau, (N_MOTORS,)).astype(np.float32)
    for i in range(N_MOTORS):
        cmd_msg.motor_cmd[i].q = float(q_des[i])
        cmd_msg.motor_cmd[i].dq = float(dq_des[i])
        cmd_msg.motor_cmd[i].kp = float(kp[i])
        cmd_msg.motor_cmd[i].kd = float(kd[i])
        cmd_msg.motor_cmd[i].tau = float(tau[i])
        cmd_msg.motor_cmd[i].mode = 1  # PD on motor


def send_cmd(publisher, cmd_msg, crc_obj):
    cmd_msg.crc = crc_obj.Crc(cmd_msg)
    publisher.Write(cmd_msg)


def force_record_path(record_name):
    record_name = os.path.basename(record_name)
    if not record_name.endswith(".npz"):
        record_name = f"{record_name}.npz"
    return os.path.join(_REPO_ROOT, "data", record_name)


# ─── Forward kinematics for wrist (replaces privileged d.xpos) ──────────────

def forward_kin_wrist_pelvis(rm: RobotModel, side: str, q27: np.ndarray) -> np.ndarray:
    """Pelvis-frame wrist position from measured joint angles.

    ``RobotModel(handless=True)`` builds a FIXED-base Pinocchio model (no free-flyer):
    nq == nv == 27 and `model.names[i+1]` for i in 0..26 matches the DDS motor order.
    So q_pinocchio = q27 directly (pelvis frame = the model's root).
    """
    import pinocchio as pin
    model = rm.model_body
    data = rm.data_body if hasattr(rm, "data_body") else model.createData()
    pin.forwardKinematics(model, data, np.asarray(q27, dtype=np.float64))
    pin.updateFramePlacements(model, data)
    frame_id = model.getFrameId(f"{side}_wrist_yaw_link")
    return data.oMf[frame_id].translation.copy()


# ─── Compute-observation proxy ──────────────────────────────────────────────

class _DataProxy:
    """Duck-typed `d` for compute_observation. Only reads qpos[3:7] (pelvis quat)
    and qvel[3:6] (pelvis omega); we provide enough of qpos/qvel for that."""

    def __init__(self, pelvis_pos, pelvis_quat, pelvis_omega, q27, dq27):
        self.qpos = np.concatenate([pelvis_pos, pelvis_quat, q27]).astype(np.float64)
        self.qvel = np.concatenate([np.zeros(3), pelvis_omega, dq27]).astype(np.float64)


def _start_key_pressed():
    import select

    try:
        ready = select.select([sys.stdin], [], [], 0.0)[0]
    except (OSError, ValueError):
        return False
    if not ready:
        return False
    try:
        return sys.stdin.read(1).lower() == "s"
    except (OSError, ValueError):
        return False


def move_to_initial_pose(
    state, publisher, cmd_msg, crc_obj, ik, *, side, arm_down, torso,
    initial_target, default_angles, leg_count, upper_n, kp_27, kd_27, dt,
):
    """Move all motors to the initial task pose gradually over 5s and hold until the user presses S."""
    print(
        "[init] moving motors to initial pose over 5s. Press S to start policy after.",
        flush=True,
    )
    old_terminal_settings = None
    if sys.stdin.isatty():
        import termios
        import tty

        old_terminal_settings = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())

    # Wait for the first state update to capture current positions
    while not state.have_low_state:
        state.new_state_event.wait(0.02)
        state.new_state_event.clear()
        
    st = state.snapshot()
    start_q = st["q"].copy()
    
    start_time = time.time()
    duration = 5.0

    arm_q_init = arm_down.copy().astype(np.float32)
    try:
        while True:
            state.new_state_event.wait(timeout=0.02)
            state.new_state_event.clear()
            st = state.snapshot()

            arm_q_init = ik.step(initial_target, dt).astype(np.float32)
            
            # Construct target state
            target_q = np.zeros(N_MOTORS, dtype=np.float32)
            target_q[:leg_count] = default_angles[:leg_count]
            
            upper_cmd = np.zeros(upper_n, dtype=np.float32)
            upper_cmd[0] = torso
            upper_cmd[ARM_SLICE[side]] = arm_q_init
            upper_cmd[ARM_SLICE["left" if side == "right" else "right"]] = arm_down
            target_q[leg_count:leg_count + upper_n] = upper_cmd[:upper_n]

            # Interpolate
            alpha = min(1.0, (time.time() - start_time) / duration)
            q_des = (1.0 - alpha) * start_q + alpha * target_q

            fill_low_cmd(cmd_msg, q_des=q_des, dq_des=0.0, kp=kp_27, kd=kd_27, tau=0.0)
            send_cmd(publisher, cmd_msg, crc_obj)

            if _start_key_pressed():
                if alpha < 1.0:
                    print("[init] S pressed early; jumping to policy.", flush=True)
                else:
                    print("[init] S pressed; starting policy.", flush=True)
                return arm_q_init
    finally:
        if old_terminal_settings is not None:
            import termios

            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_terminal_settings)


# ─── Main control loop ─────────────────────────────────────────────────────

def run_manip_dds(state, publisher, cmd_msg, crc_obj, config, rm, policy, encoder, *,
                  side, xyz_at, total_s, payload_kg, settle_s, load_ramp_s, torso,
                  no_encode, use_force_estimator, payload_at=None, label=""):
    """One DDS-driven episode. Returns (metrics_dict, traj_dict)."""
    import collections

    # ── Wait for first state + settle the free base under gravity ──
    print("[ctrl] waiting for first LowState / SportModeState ...", flush=True)
    if not state.new_state_event.wait(timeout=5.0):
        raise TimeoutError("No LowState received in 5 s.")
    state.new_state_event.clear()
    while not state.have_high_state:
        state.new_state_event.wait(0.1)
        state.new_state_event.clear()
    print(f"[ctrl] first state received (tick={state.tick}). Settling 0.5 s ...", flush=True)

    # Send a soft-hold for 0.5 s while the base settles + watchdog stays happy
    settle_end = time.time() + 0.5
    while time.time() < settle_end:
        state.new_state_event.wait(0.1); state.new_state_event.clear()
        st = state.snapshot()
        fill_low_cmd(cmd_msg, q_des=st["q"], dq_des=0.0, kp=80.0, kd=2.0, tau=0.0)
        send_cmd(publisher, cmd_msg, crc_obj)

    dt = SIM_DT
    decim = POLICY_DECIM

    # ── Setup IK + bookkeeping ──
    leg_count = config["num_actions"]
    upper_n = int(config.get("h12_ctrl_count", N_MOTORS)) - leg_count
    arm_down = np.asarray(config.get("_arm_down"), dtype=np.float32)
    ik = ArmIK(rm, side, arm_down, torso)
    arm_q_hold = arm_down.copy().astype(np.float32)

    # PD gains as 27-vectors (legs use config['kps']/['kds'], upper body uses YAML gains).
    kp_27 = np.zeros(N_MOTORS, dtype=np.float32)
    kd_27 = np.zeros(N_MOTORS, dtype=np.float32)
    kp_27[:leg_count] = config["kps"]
    kd_27[:leg_count] = config["kds"]
    kps_arms = config.get("kps_arms", np.full(upper_n, 100.0, dtype=np.float32))
    kds_arms = config.get("kds_arms", np.full(upper_n, 5.0, dtype=np.float32))
    kp_27[leg_count:leg_count + upper_n] = kps_arms[:upper_n]
    kd_27[leg_count:leg_count + upper_n] = kds_arms[:upper_n]

    arm_q_hold = move_to_initial_pose(
        state, publisher, cmd_msg, crc_obj, ik,
        side=side,
        arm_down=arm_down,
        torso=torso,
        initial_target=np.asarray(xyz_at(0.0), dtype=np.float32),
        default_angles=np.asarray(config["default_angles"], dtype=np.float32),
        leg_count=leg_count,
        upper_n=upper_n,
        kp_27=kp_27,
        kd_27=kd_27,
        dt=dt,
    )

    # ── Snapshot the nominal pelvis pose AFTER initialization / user start ──
    st0 = state.snapshot()
    p0, q0 = pelvis_from_imu(st0["imu_pos"], st0["imu_quat"], st0["q"][TORSO_MOTOR_IDX])
    R0 = quat2R(q0)
    print(f"[ctrl] nominal pelvis pose: p0={p0.round(3)}  q0={q0.round(3)}", flush=True)

    policy_joints = int(config.get("policy_num_joints", N_MOTORS))
    cmd_3 = config["cmd_init"].copy()
    height_cmd = float(config["height_cmd"])
    action = np.zeros(leg_count, dtype=np.float32)
    target_dof_pos = config["default_angles"].copy()

    obs_hist = collections.deque(maxlen=config["obs_history_len"])
    for _ in range(config["obs_history_len"]):
        obs_hist.append(np.zeros(76, dtype=np.float32))  # 76 = single_obs_dim
    z_hist = np.zeros((3, RMA_LATENT_DIM), dtype=np.float32)
    g = np.array([0.0, 0.0, -9.81])

    n_steps_task = int(total_s / dt)

    traj = {"t": [], "world": [], "cmd_world": [], "bpos": [], "bquat": [], "load": [], "z": []}
    force_record = {"t": [], "left_estimated_force": [], "right_estimated_force": []}
    ssq_base = ssq_ee = ssq_eeb = 0.0
    max_ee = max_tilt = 0.0
    ss_ee = 0.0
    n_acc = ss_n = 0
    fell = False
    ss_start = max(0, n_steps_task - int(round(1.0 / dt)))
    e_ee = e_ee_b = 0.0

    # ── Main loop (event-driven on LowState arrival ≈ 200 Hz) ──
    # Runs the waypoint task for n_steps_task ticks, then keeps running so the
    # FAME policy keeps balancing the legs and the arm tracks the final waypoint
    # (xyz_at clamps for t > total_s). Ctrl-C exits.
    last_loop_warn_t = 0.0
    task_metrics = None
    import itertools

    def _snapshot_metrics():
        n_ee = max(1, len(traj["t"]))
        return {
            "ee_rmse": float(np.sqrt(ssq_ee / n_ee)),
            "ee_max": max_ee,
            "ee_ss": ss_ee / max(1, ss_n),
            "ee_b_rmse": float(np.sqrt(ssq_eeb / n_ee)),
            "base_rmse": float(np.sqrt(ssq_base / max(1, n_acc))),
            "tilt_max": max_tilt,
            "fell": int(fell),
            "traj": {k: np.array(v) for k, v in traj.items()},
            "force_record": {k: np.array(v) for k, v in force_record.items()},
        }

    try:
        for step in itertools.count():
            if not state.new_state_event.wait(timeout=0.05):
                # Sim stalled — keep watchdog happy with a hold-current-pose command.
                now = time.time()
                if now - last_loop_warn_t > 1.0:
                    print(f"[ctrl] WARN: LowState stalled at step {step}", flush=True)
                    last_loop_warn_t = now
                st = state.snapshot()
                fill_low_cmd(cmd_msg, q_des=st["q"], dq_des=0.0, kp=kp_27, kd=kd_27, tau=0.0)
                send_cmd(publisher, cmd_msg, crc_obj)
                continue
            state.new_state_event.clear()

            st = state.snapshot()
            t = step * dt
            pelvis_c, pelvis_q = pelvis_from_imu(st["imu_pos"], st["imu_quat"], st["q"][TORSO_MOTOR_IDX])
            Rp_c = quat2R(pelvis_q)

            # ── Commanded payload (always computed — used by traj["load"] and as the
            #     privileged fallback for the encoder force input below) ──
            if payload_at is not None:
                kg_now = float(payload_at(t))
            else:
                lr = smoothstep((t - settle_s) / max(load_ramp_s, 1e-6))
                kg_now = lr * payload_kg
            F_commanded = kg_now * g   # world-frame, sign = force ON the wrist

            # ── Force fed to the encoder ────────────────────────────────────────────
            # use_force_estimator (from YAML) selects the source. Default is the
            # privileged commanded force.
            #
            # Expected contract:
            #   - Returns ONE 3-vector per hand, WORLD frame, Newtons
            #   - Sign: force ON the wrist  (e.g. 2 kg hanging → ≈[0, 0, -19.6])
            #   - Magnitudes > 30 N are direction-preserving clipped INSIDE
            #     build_et_mujoco — no need to clip here.
            #
            # NOTE on sim: the controller does NOT apply `xfrc_applied` to the sim
            # (no DDS topic for it), so the sim's wrist physically feels no load,
            # joint torques don't reflect the payload, and an inverse-dynamics
            # estimator in sim will read ≈ 0 regardless of the commanded payload.
            # Validate the estimator on the real robot, or first add an xfrc DDS
            # topic to h1_mujoco.
            if use_force_estimator:
                robot_model = rm
                try:
                    left_wrench = robot_model.get_frame_wrench(
                        "left_wrist_yaw_link", q=st["q"], tau=st["tau_est"],
                        imu_quat=st["imu_quat"])
                    right_wrench = robot_model.get_frame_wrench(
                        "right_wrist_yaw_link", q=st["q"], tau=st["tau_est"],
                        imu_quat=st["imu_quat"])
                    # Match deploy_real's sign convention: force ON wrist.
                    ef_left = (-left_wrench[:3]).astype(np.float32)
                    ef_right = (-right_wrench[:3]).astype(np.float32)
                except Exception as exc:
                    if step == 0:
                        print(f"[force_estimator] failed: {exc}; falling back to zeros", flush=True)
                    ef_left = np.zeros(3, np.float32)
                    ef_right = np.zeros(3, np.float32)
            else:
                # PRIVILEGED default: commanded payload weight, on the active side(s).
                ef_left  = F_commanded.astype(np.float32) if side in ("left", "both") else np.zeros(3, np.float32)
                ef_right = F_commanded.astype(np.float32) if side in ("right", "both") else np.zeros(3, np.float32)
            force_record["t"].append(t)
            force_record["left_estimated_force"].append(ef_left.copy())
            force_record["right_estimated_force"].append(ef_right.copy())

            # ── IK every `decim` ticks: world-fixed target → current base frame ──
            # After the task (t > total_s), xyz_at clamps to the final waypoint, so the
            # IK just keeps the arm at the place pose while the legs balance.
            if step % decim == 0:
                world_tgt = R0 @ xyz_at(t) + p0
                base_tgt = Rp_c.T @ (world_tgt - pelvis_c)
                arm_q_hold = ik.step(base_tgt, dt * decim).astype(np.float32)

            # ── Build the 27-DOF target vector ──
            q_des = np.zeros(N_MOTORS, dtype=np.float32)
            q_des[:leg_count] = target_dof_pos[:leg_count]
            upper_cmd = np.zeros(15, dtype=np.float32)
            upper_cmd[0] = torso
            upper_cmd[ARM_SLICE[side]] = arm_q_hold
            upper_cmd[ARM_SLICE["left" if side == "right" else "right"]] = arm_down
            q_des[leg_count:leg_count + 15] = upper_cmd

            fill_low_cmd(cmd_msg, q_des=q_des, dq_des=0.0, kp=kp_27, kd=kd_27, tau=0.0)
            send_cmd(publisher, cmd_msg, crc_obj)

            # ── Metrics: pelvis drift / tilt / fall detection ──
            e_base = float(np.linalg.norm(pelvis_c[:2] - p0[:2]))
            tilt = float(np.degrees(np.arccos(np.clip(-get_gravity_orientation(pelvis_q)[2], -1, 1))))
            ssq_base += e_base ** 2
            max_tilt = max(max_tilt, tilt)
            n_acc += 1
            if pelvis_c[2] < HEIGHT_THRESHOLD or tilt > TILT_DEG_THRESHOLD:
                fell = True

            if step % decim == 0:
                # ee_w from FK on measured joints (replaces privileged d.xpos[manip_eid]).
                ee_b = forward_kin_wrist_pelvis(rm, side, st["q"])
                ee_w = Rp_c @ ee_b + pelvis_c
                cmd_world = R0 @ xyz_at(t) + p0
                dist_world = Rp_c @ xyz_at(t) + pelvis_c
                e_ee = float(np.linalg.norm(ee_w - cmd_world))
                e_ee_b = float(np.linalg.norm(dist_world - cmd_world))
                ssq_ee += e_ee ** 2
                ssq_eeb += e_ee_b ** 2
                max_ee = max(max_ee, e_ee)
                if step >= ss_start:
                    ss_ee += e_ee
                    ss_n += 1

                traj["t"].append(t)
                traj["world"].append(ee_w.copy())
                traj["cmd_world"].append(cmd_world)
                traj["bpos"].append(pelvis_c.copy())
                traj["bquat"].append(pelvis_q.copy())
                traj["load"].append(kg_now)

                d_proxy = _DataProxy(pelvis_c, pelvis_q, st["imu_gyro"], st["q"], st["dq"])
                single_obs, _ = compute_observation(d_proxy, config, action, cmd_3,
                                                    height_cmd, policy_joints,
                                                    qj=st["q"], dqj=st["dq"])
                obs_hist.append(single_obs)

                e_t = build_et_mujoco(d_proxy.qpos,
                                      np.zeros(3) if no_encode else ef_left,
                                      np.zeros(3) if no_encode else ef_right,
                                      leg_count, policy_joints, st["q"])
                with torch.no_grad():
                    z = encoder(torch.from_numpy(e_t).unsqueeze(0).float()).numpy().squeeze()
                z_hist[1:] = z_hist[:-1]
                z_hist[0] = z
                traj["z"].append(float(np.linalg.norm(z)))

                z_flat = np.flip(z_hist, 0).flatten().astype(np.float32)
                proprio = np.concatenate(list(obs_hist), axis=0)
                actor_obs = np.concatenate([proprio, z_flat], axis=0).astype(np.float32)
                action = policy(torch.from_numpy(actor_obs).unsqueeze(0)).detach().numpy().squeeze()
                target_dof_pos = clip_policy_action(action, config)

            # ── Task-end snapshot: freeze metrics + print summary; loop keeps running ──
            if step + 1 == n_steps_task and task_metrics is None:
                task_metrics = _snapshot_metrics()
                print(
                    f"  e^W_ee rmse={task_metrics['ee_rmse'] * 100:5.1f}cm "
                    f"max={task_metrics['ee_max'] * 100:5.1f}cm | "
                    f"e^W_base={task_metrics['ee_b_rmse'] * 100:5.1f}cm "
                    f"pelvis drift={task_metrics['base_rmse'] * 100:4.1f}cm "
                    f"tilt={task_metrics['tilt_max']:4.1f} fell={task_metrics['fell']}",
                    flush=True,
                )
                print("[ctrl] task complete — FAME keeps balancing; press Ctrl-C to release.",
                      flush=True)
    except KeyboardInterrupt:
        print("\n[ctrl] released by user", flush=True)

    # If interrupted before the task finished, snapshot whatever we have.
    if task_metrics is None:
        task_metrics = _snapshot_metrics()
    return task_metrics


def shutdown_soft(state, publisher, cmd_msg, crc_obj, n_ticks=20):
    """Soft-damping hold for ~100 ms before exit, so the robot bleeds energy
    instead of going limp via the sim watchdog."""
    for _ in range(n_ticks):
        st = state.snapshot()
        fill_low_cmd(cmd_msg, q_des=st["q"], dq_des=0.0, kp=0.0, kd=2.0, tau=0.0)
        send_cmd(publisher, cmd_msg, crc_obj)
        time.sleep(0.005)


# ─── CLI / main ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="DDS-decoupled manip_ik_demo")
    p.add_argument("--config", default=os.path.join(_SCRIPT_DIR, "h12_fame.yaml"))
    p.add_argument("--manip_yaml", default=os.path.join(_SCRIPT_DIR, "single_arm_manip.yaml"))
    p.add_argument("--task", default="right_hand_manip",
                   choices=("right_hand_manip", "left_hand_manip"))
    p.add_argument("--payload_kg", type=float, default=None, help="Override YAML payload")
    p.add_argument("--no_encode", action="store_true",
                   help="Zero the encoder force input (the 'no-FAME' condition)")
    p.add_argument("--adapt", action="store_true",
                   help="Apply an on-the-fly payload schedule (sudden steps in 0-3 kg). "
                        "Saves a single-condition adapt plot at task end.")
    p.add_argument("--save_plot", default=None,
                   help="Path for the adapt plot PNG. Defaults to "
                        "simulation_exp/figures/manip_{side}_adapt_dds_{cond}.png.")
    p.add_argument("--save_traj", default=None,
                   help="Path to write the run's traj dict as .npz "
                        "(for later use with deploy/compare_plot_dds.py).")
    p.add_argument("--net", default=None,
                   help="DDS network interface (default: SDK picks one; 'lo' for local sim)")
    p.add_argument("--dds_id", type=int, default=0)
    p.add_argument("--wait_state_s", type=float, default=5.0)
    args = p.parse_args()

    # ── Load configs ──
    mc = yaml.safe_load(open(args.manip_yaml))
    task = mc[args.task]
    side = task["manip"]
    arm_down = np.asarray(mc["arm_down"], dtype=np.float32)
    torso = float(mc.get("torso", -0.35))
    payload = args.payload_kg if args.payload_kg is not None else float(task.get("payload_kg", 1.0))
    settle_s = 1.0
    seg_s = float(mc.get("seg_time_s", 1.5))
    hold_s = float(mc.get("hold_s", 0.8))
    ramp_s = float(mc.get("load_ramp_s", 0.8))
    oc = float(mc.get("orientation_cost", 0.0)) if mc.get("track_orientation") else 0.0

    config = load_config(args.config)
    cdir = config["_config_dir"]
    for k in ("policy_path", "encoder_path"):
        if config.get(k) and not os.path.isabs(config[k]):
            config[k] = os.path.normpath(os.path.join(cdir, config[k]))
    config["_arm_down"] = arm_down
    low_cmd_topic = config.get("lowcmd_topic", DEFAULT_LOW_CMD_TOPIC)
    low_state_topic = config.get("lowstate_topic", DEFAULT_LOW_STATE_TOPIC)
    high_state_topic = config.get("highstate_topic", DEFAULT_HIGH_STATE_TOPIC)
    force_config = config.get("force", {})
    use_force_estimator = bool(force_config.get("use_force_estimator", DEFAULT_FORCE_ESTIMATOR_ENABLED))
    record_force = bool(force_config.get("record_force", False))
    force_record_name = force_config.get("record_name", "force_record")

    # ── Pinocchio model + policy + encoder ──
    robot_model = RobotModel(os.path.join(_REPO_ROOT, "h1_2/h1_2_handless.urdf"), handless=True)
    robot_model.data_body = robot_model.model_body.createData()  # cache for FK

    policy = torch.jit.load(config["policy_path"])
    policy.eval()
    encoder = EnvFactorEncoder(EnvFactorEncoderCfg())
    encoder.load_state_dict(torch.load(config["encoder_path"], map_location="cpu", weights_only=True))
    encoder.eval()

    # IK setup print
    sols, resid = solve_arm_waypoints(robot_model, side, arm_down, torso, task["waypoints"],
                                      orientation_cost=oc)
    print(f"[IK] task={args.task} side={side} payload={payload}kg")
    for wp, r in zip(task["waypoints"], resid):
        flag = "" if r < 0.01 else "  <-- UNREACHABLE (residual high)"
        print(f"   {wp['name']:6s} xyz={wp['xyz']}  IK residual={r * 1000:5.1f}mm{flag}")
    xyz_at, total = build_xyz_schedule(task["waypoints"], settle_s, seg_s, hold_s)

    # ── DDS init ──
    print(f"[dds] ChannelFactoryInitialize(id={args.dds_id}, networkInterface={args.net!r})")
    ChannelFactoryInitialize(args.dds_id, args.net) if args.net else ChannelFactoryInitialize(args.dds_id)
    print(
        f"[dds] topics: lowcmd={low_cmd_topic} lowstate={low_state_topic} "
        f"highstate={high_state_topic} force_estimator={use_force_estimator} "
        f"record_force={record_force}",
        flush=True,
    )

    robot_model.init_subscriber(low_state_topic=low_state_topic)

    state = DDSState()
    low_state_sub = ChannelSubscriber(low_state_topic, LowStateHG)
    low_state_sub.Init(make_low_state_callback(state), 10)
    high_state_sub = ChannelSubscriber(high_state_topic, SportModeState_)
    high_state_sub.Init(make_high_state_callback(state), 10)

    cmd_msg = unitree_hg_msg_dds__LowCmd_()
    publisher = ChannelPublisher(low_cmd_topic, LowCmdHG)
    publisher.Init()
    crc_obj = CRC()

    # ── Optional: on-the-fly payload schedule (--adapt) ──
    payload_at = None
    step_times = None
    if args.adapt:
        payload_at, step_times = build_payload_profile(settle_s)
        print(f"[adapt] on-the-fly payload steps at t={[round(s, 1) for s in step_times]}s "
              f"(within 0-3kg / 30 N trained limit)")

    # ── Run ──
    cond = "no-FAME" if args.no_encode else "FAME"
    print(f"\n================ {args.task}: {cond}  payload={payload}kg "
          f"{'(adapt)' if args.adapt else ''} ================")
    label = f"{cond} ({args.task}, {payload}kg)"
    task_metrics = None
    try:
        task_metrics = run_manip_dds(
            state, publisher, cmd_msg, crc_obj, config, robot_model, policy, encoder,
            side=side, xyz_at=xyz_at, total_s=total, payload_kg=payload,
            settle_s=settle_s, load_ramp_s=ramp_s, torso=torso,
            no_encode=args.no_encode, use_force_estimator=use_force_estimator,
            payload_at=payload_at, label=label,
        )
    except KeyboardInterrupt:
        print("\n[ctrl] interrupted during task")
    finally:
        # Save the adapt plot before the soft-damping exit (we have task_metrics["traj"]).
        if args.adapt and task_metrics is not None and len(task_metrics["traj"]["t"]) > 0:
            out_path = args.save_plot or os.path.join(
                _REPO_ROOT, "simulation_exp", "figures",
                f"manip_{side}_adapt_dds_{cond.replace('-', '_').lower()}.png",
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plot_adapt_single(task_metrics["traj"], step_times, args.task, out_path,
                              cond_label=cond)
        # Save the traj as .npz (for offline FAME-vs-no-FAME comparison via
        # deploy/compare_plot_dds.py). Also stash a few task-level scalars.
        if args.save_traj and task_metrics is not None and len(task_metrics["traj"]["t"]) > 0:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_traj)) or ".",
                        exist_ok=True)
            np.savez(
                args.save_traj,
                **task_metrics["traj"],
                cond=cond,
                task=args.task,
                payload_kg=float(payload),
                adapt=bool(args.adapt),
                step_times=np.asarray(step_times if step_times is not None else [],
                                      dtype=np.float64),
                ee_rmse=task_metrics["ee_rmse"],
                ee_max=task_metrics["ee_max"],
                ee_b_rmse=task_metrics["ee_b_rmse"],
                base_rmse=task_metrics["base_rmse"],
                tilt_max=task_metrics["tilt_max"],
                fell=task_metrics["fell"],
            )
            print(f"saved traj -> {args.save_traj}")
        if record_force and task_metrics is not None:
            record = task_metrics["force_record"]
            out_path = force_record_path(force_record_name)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.savez(
                out_path,
                force_time=record["t"],
                left_estimated_force=record["left_estimated_force"],
                right_estimated_force=record["right_estimated_force"],
            )
            print(f"saved force record -> {out_path}")
        print("[ctrl] shutting down (soft damping)")
        shutdown_soft(state, publisher, cmd_msg, crc_obj)


if __name__ == "__main__":
    main()
