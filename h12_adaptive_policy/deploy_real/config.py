import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from unitree_sdk2py.core.channel import ChannelFactoryInitialize


def _channel_factory_settings(config: dict[str, Any]) -> tuple[int, str | None]:
    '''Get DDS channel factory settings from environment and config'''
    env_domain = os.environ.get('ROS_DOMAIN_ID')
    network = config['network']
    domain_id = (
        int(env_domain)
        if env_domain is not None
        else int(network['domain_id'])
    )
    return domain_id, network['interface']


def init_channel_factory(config: dict[str, Any]) -> None:
    '''Initialize the DDS channel factory from environment and config'''
    domain_id, interface = _channel_factory_settings(config)
    if interface:
        ChannelFactoryInitialize(domain_id, interface)
    else:
        ChannelFactoryInitialize(domain_id)


def init_channel_factory_guard(config: dict[str, Any]) -> None:
    '''Confirm before initializing the real robot DDS domain'''
    domain_id, _ = _channel_factory_settings(config)
    if domain_id == 0:
        try:
            confirmation = input(
                'WARNING: ROS_DOMAIN_ID=0 -> DDS domain 0 is the REAL ROBOT '
                'command bus.\n'
                '         Nodes will publish/subscribe on the live robot.\n'
                'Proceed on DDS domain 0 (real robot)? [y/N] '
            )
        except EOFError:
            confirmation = ''
        if confirmation.strip().lower() != 'y':
            raise SystemExit('DDS channel factory initialization cancelled')
    init_channel_factory(config)


class Config:
    def __init__(self, file_path) -> None:
        self.project_root = Path(__file__).resolve().parent.parent.parent

        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.data = config

            self.control_dt = config["control_dt"]

            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]

            self.weak_motor = []
            if "weak_motor" in config:
                self.weak_motor = config["weak_motor"]

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]

            policy_path = Path(config["policy_path"])
            if not policy_path.is_absolute():
                policy_path = self.project_root / policy_path
            self.policy_path = str(policy_path)

            # Optional RMA encoder path (for RMA real deploy)
            encoder_path_cfg = config.get("encoder_path")
            if encoder_path_cfg:
                encoder_path = Path(encoder_path_cfg)
                if not encoder_path.is_absolute():
                    encoder_path = self.project_root / encoder_path
                self.encoder_path = str(encoder_path)
            else:
                self.encoder_path = None

            self.leg_joint2motor_idx = config["leg_joint2motor_idx"]
            self.kps = config.get("kps") or config["kps_legs"]
            self.kds = config.get("kds") or config["kds_legs"]
            self.default_angles = np.array(config.get("default_angles") or config["default_angles_legs"], dtype=np.float32)

            self.arm_waist_joint2motor_idx = config["arm_waist_joint2motor_idx"]
            self.arm_waist_kps = config["arm_waist_kps"]
            self.arm_waist_kds = config["arm_waist_kds"]
            self.arm_waist_target = np.array(config["arm_waist_target"], dtype=np.float32)

            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
            # self.max_cmd = np.array(config["max_cmd"], dtype=np.float32)
            self.num_dofs = config["num_dofs"]
            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]
            self.obs_history_len = config["obs_history_len"]
            self.single_obs_dim = config.get("single_obs_dim", 76)
            self.cmd_init = np.array(config.get("cmd_init", [0.0, 0.0, 0.0]), dtype=np.float32)
            self.height_cmd = float(config.get("height_cmd", 1.00))

            # Optional RMA extrinsics for hands (world-frame forces), default zeros
            self.left_hand_force = np.array(config.get("left_hand_force", [0.0, 0.0, 0.0]), dtype=np.float32)
            self.right_hand_force = np.array(config.get("right_hand_force", [0.0, 0.0, 0.0]), dtype=np.float32)
            self.no_encode = bool(config.get("no_encode", False))

            self.legs_motor_pos_lower_limit_list = config["legs_motor_pos_lower_limit_list"]
            self.legs_motor_pos_upper_limit_list = config["legs_motor_pos_upper_limit_list"]
