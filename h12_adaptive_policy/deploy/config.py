"""Centralized deploy/task YAML loading and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import yaml


DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
ARM_SLICE = {"left": slice(1, 8), "right": slice(8, 15)}


@dataclass(frozen=True)
class ForceConfig:
    use_force_estimator: bool
    record_force: bool
    record_name: str


@dataclass(frozen=True)
class StartupConfig:
    initial_move_duration_s: float
    preposition_duration_s: float
    preposition_error_tolerance_m: float
    preposition_timeout_s: float


@dataclass(frozen=True)
class UpperBodyDefaults:
    torso: float
    left_arm: np.ndarray
    right_arm: np.ndarray


@dataclass(frozen=True)
class ManipTask:
    name: str
    side: str
    payload_kg: float
    waypoints: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ManipConfig:
    path: str
    data: dict[str, Any]
    ee_frame: dict[str, str]
    seg_time_s: float
    hold_s: float
    load_ramp_s: float
    position_cost: float
    orientation_cost: float
    track_orientation: bool
    arm_down: np.ndarray | None

    def task(self, name: str) -> ManipTask:
        if name not in self.data:
            raise KeyError(f"Task {name!r} not found in {self.path}")
        raw = self.data[name]
        side = raw.get("manip")
        if side not in ("left", "right", "both"):
            raise ValueError(f"Task {name!r} has invalid manip side {side!r}")
        waypoints = tuple(raw.get("waypoints", ()))
        if side in ("left", "right"):
            _validate_waypoints(waypoints, f"{name}.waypoints")
        return ManipTask(
            name=name,
            side=side,
            payload_kg=float(raw.get("payload_kg", 1.0)),
            waypoints=waypoints,
        )


def resolve_config_path(config_path: str) -> str:
    """Resolve YAML paths, accepting absolute/relative paths or deploy filenames."""
    config_path = os.path.expanduser(config_path)
    candidates = [config_path]
    if not os.path.splitext(config_path)[1]:
        candidates.append(f"{config_path}.yaml")

    for candidate in candidates:
        if os.path.isabs(candidate):
            resolved = candidate
        elif os.path.dirname(candidate):
            resolved = os.path.abspath(candidate)
        else:
            resolved = os.path.join(DEPLOY_DIR, candidate)
        if os.path.isfile(resolved):
            return resolved

    tried = ", ".join(candidates)
    raise FileNotFoundError(
        f"Config not found: {tried}. Pass a full path/relative path or a filename in {DEPLOY_DIR}."
    )


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML {path} must contain a mapping at the top level")
    return data


def load_config(config_path: str) -> dict[str, Any]:
    """Load deploy YAML into the dict shape expected by existing scripts."""
    config_path = resolve_config_path(config_path)
    config = load_yaml(config_path)
    config["_config_path"] = config_path
    config["_config_dir"] = os.path.dirname(config_path)

    if "upper_body_pd_gains" in config:
        gains = config["upper_body_pd_gains"]
        config["kps_arms"] = np.asarray(gains["kp"], dtype=np.float32)
        config["kds_arms"] = np.asarray(gains["kd"], dtype=np.float32)

    for key in (
        "kps",
        "kds",
        "kps_arms",
        "kds_arms",
        "default_lower_angles",
        "default_upper_angles",
        "cmd_scale",
        "cmd_init",
        "left_hand_force",
        "right_hand_force",
        "legs_motor_pos_lower_limit_list",
        "legs_motor_pos_upper_limit_list",
    ):
        if key in config:
            config[key] = np.asarray(config[key], dtype=np.float32)

    _resolve_relative_paths(config, ("policy_path", "xml_path", "encoder_path"))
    _validate_deploy_config(config)
    config["force_config"] = ForceConfig(**config["force"])
    config["startup_config"] = StartupConfig(**config["startup"])
    config["upper_body_defaults"] = upper_body_defaults(config)
    return config


def load_manip_config(manip_path: str) -> ManipConfig:
    path = resolve_config_path(manip_path)
    data = load_yaml(path)
    if "ee_frame" not in data:
        raise ValueError(f"{path}: missing required field ee_frame")
    mc = ManipConfig(
        path=path,
        data=data,
        ee_frame=dict(data["ee_frame"]),
        seg_time_s=float(data.get("seg_time_s", 1.5)),
        hold_s=float(data.get("hold_s", 0.8)),
        load_ramp_s=float(data.get("load_ramp_s", 0.8)),
        position_cost=float(data.get("position_cost", 50.0)),
        orientation_cost=float(data.get("orientation_cost", 0.0)),
        track_orientation=bool(data.get("track_orientation", False)),
        arm_down=_optional_array(data, "arm_down", (7,)),
    )
    return mc


def upper_body_defaults(config: dict[str, Any]) -> UpperBodyDefaults:
    upper = _require_array(config, "default_upper_angles", (15,))
    return UpperBodyDefaults(
        torso=float(upper[0]),
        left_arm=upper[ARM_SLICE["left"]].copy(),
        right_arm=upper[ARM_SLICE["right"]].copy(),
    )


def _resolve_relative_paths(config: dict[str, Any], keys: tuple[str, ...]) -> None:
    config_dir = config["_config_dir"]
    for key in keys:
        value = config.get(key)
        if value and isinstance(value, str) and not os.path.isabs(value):
            config[key] = os.path.normpath(os.path.join(config_dir, value))


def _require_array(config: dict[str, Any], key: str, shape: tuple[int, ...]) -> np.ndarray:
    if key not in config:
        raise ValueError(f"{config.get('_config_path', '<config>')}: missing required field {key}")
    arr = np.asarray(config[key], dtype=np.float32)
    if arr.shape != shape:
        raise ValueError(f"{config.get('_config_path', '<config>')}: {key} must have shape {shape}, got {arr.shape}")
    return arr


def _optional_array(data: dict[str, Any], key: str, shape: tuple[int, ...]) -> np.ndarray | None:
    if key not in data:
        return None
    arr = np.asarray(data[key], dtype=np.float32)
    if arr.shape != shape:
        raise ValueError(f"{key} must have shape {shape}, got {arr.shape}")
    return arr


def _validate_deploy_config(config: dict[str, Any]) -> None:
    required_scalars = (
        "policy_path",
        "encoder_path",
        "xml_path",
        "policy_num_joints",
        "h12_ctrl_count",
        "simulation_duration",
        "simulation_dt",
        "control_decimation",
        "ang_vel_scale",
        "dof_pos_scale",
        "dof_vel_scale",
        "action_scale",
        "num_actions",
        "num_obs",
        "obs_history_len",
        "height_cmd",
    )
    for key in required_scalars:
        if key not in config:
            raise ValueError(f"{config['_config_path']}: missing required field {key}")

    _require_array(config, "kps", (12,))
    _require_array(config, "kds", (12,))
    _require_array(config, "kps_arms", (15,))
    _require_array(config, "kds_arms", (15,))
    _require_array(config, "default_lower_angles", (12,))
    _require_array(config, "default_upper_angles", (15,))
    _require_array(config, "cmd_scale", (3,))
    _require_array(config, "cmd_init", (3,))
    _require_array(config, "left_hand_force", (3,))
    _require_array(config, "right_hand_force", (3,))
    _require_array(config, "legs_motor_pos_lower_limit_list", (12,))
    _require_array(config, "legs_motor_pos_upper_limit_list", (12,))

    if "force" not in config:
        raise ValueError(f"{config['_config_path']}: missing required field force")
    force = config["force"]
    for key in ("use_force_estimator", "record_force", "record_name"):
        if key not in force:
            raise ValueError(f"{config['_config_path']}: missing required field force.{key}")

    if "startup" not in config:
        raise ValueError(f"{config['_config_path']}: missing required field startup")
    startup = config["startup"]
    for key in ("initial_move_duration_s", "preposition_duration_s", "preposition_error_tolerance_m", "preposition_timeout_s"):
        if key not in startup:
            raise ValueError(f"{config['_config_path']}: missing required field startup.{key}")


def _validate_waypoints(waypoints: tuple[dict[str, Any], ...], label: str) -> None:
    if not waypoints:
        raise ValueError(f"{label} must contain at least one waypoint")
    for i, wp in enumerate(waypoints):
        if "xyz" not in wp:
            raise ValueError(f"{label}[{i}] missing xyz")
        xyz = np.asarray(wp["xyz"], dtype=np.float32)
        if xyz.shape != (3,):
            raise ValueError(f"{label}[{i}].xyz must have shape (3,), got {xyz.shape}")
