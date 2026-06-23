"""
Inspect MuJoCo model structure: map qpos/ctrl indices to joint/actuator names.

This script helps debug mismatches between MuJoCo and Pinocchio definitions
by showing exactly which qpos indices and ctrl indices correspond to which
joints and actuators.

Usage:
  python inspect_mujoco_state.py
  python inspect_mujoco_state.py --config ../deploy/h12_fame.yaml
  python inspect_mujoco_state.py --xml /path/to/model.xml
"""

import sys
import os
import argparse
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_DEPLOY_DIR = os.path.join(_REPO_ROOT, 'h12_adaptive_policy', 'deploy')

import mujoco

def resolve_config_path(config_path):
    """Resolve absolute/relative config paths or filenames from deploy/."""
    config_path = os.path.expanduser(config_path)
    candidates = [config_path]
    if not os.path.splitext(config_path)[1]:
        candidates.append(f'{config_path}.yaml')

    for candidate in candidates:
        if os.path.isabs(candidate):
            resolved = candidate
        elif os.path.dirname(candidate):
            resolved = os.path.abspath(candidate)
        else:
            resolved = os.path.join(_DEPLOY_DIR, candidate)
        if os.path.isfile(resolved):
            return resolved

    tried = ', '.join(candidates)
    raise FileNotFoundError(
        f"Config not found: {tried}. Pass a full path/relative path or a filename in {_DEPLOY_DIR}."
    )

def load_config(config_path):
    """Load YAML config file."""
    import yaml
    config_path = resolve_config_path(config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['_config_path'] = config_path
    config['_config_dir'] = os.path.dirname(config_path)
    return config

def inspect_model(xml_path):
    """Load MuJoCo model and inspect qpos/ctrl mappings."""
    print(f"\n{'='*80}")
    print(f"Inspecting MuJoCo Model: {xml_path}")
    print(f"{'='*80}\n")

    m = mujoco.MjModel.from_xml_path(xml_path)

    # Create sample data
    d = mujoco.MjData(m)

    print(f"Total qpos dimension: {m.nq}")
    print(f"Total qvel dimension: {m.nv}")
    print(f"Total ctrl dimension: {m.nu}")

    # =========================================================================
    # QPOS Mapping
    # =========================================================================
    print(f"\n{'='*80}")
    print("QPOS MAPPING (Generalized Coordinates)")
    print(f"{'='*80}\n")

    # Analyze bodies and joints
    qpos_idx = 0
    print(f"{'Index':<8} {'Type':<12} {'Name':<40} {'Dim':<5} {'Range'}")
    print("-" * 100)

    # Base state (free floating body)
    print(f"{qpos_idx:<8} {'FREE':<12} {'base_position (x, y, z)':<40} {3:<5} {'[0, 2]'}")
    qpos_idx = 3
    print(f"{qpos_idx:<8} {'BALL':<12} {'base_quaternion (w, x, y, z)':<40} {4:<5} {'[3, 6]'}")
    qpos_idx = 7

    body_qpos_map = {}
    for body_id in range(m.nbody):
        body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id)

        # Find joints associated with this body
        for joint_id in range(m.njnt):
            joint = m.jnt_bodyid[joint_id]
            if joint == body_id:
                joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                joint_type = m.jnt_type[joint_id]

                # Determine dimension based on joint type
                if joint_type == mujoco.mjtJoint.mjJNT_BALL:
                    dim = 4  # quaternion
                    type_str = "BALL"
                elif joint_type == mujoco.mjtJoint.mjJNT_HINGE or joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
                    dim = 1
                    type_str = "HINGE" if joint_type == mujoco.mjtJoint.mjJNT_HINGE else "SLIDE"
                else:
                    dim = 0
                    type_str = "??"

                if dim > 0:
                    range_str = f"[{qpos_idx}, {qpos_idx + dim - 1}]"
                    print(f"{qpos_idx:<8} {type_str:<12} {joint_name:<40} {dim:<5} {range_str}")
                    body_qpos_map[body_name] = {
                        'joint_name': joint_name,
                        'type': type_str,
                        'qpos_start': qpos_idx,
                        'qpos_end': qpos_idx + dim - 1,
                        'dim': dim
                    }
                    qpos_idx += dim

    # =========================================================================
    # CTRL Mapping (Actuators)
    # =========================================================================
    print(f"\n{'='*80}")
    print("CTRL MAPPING (Actuators/Motors)")
    print(f"{'='*80}\n")

    print(f"{'Ctrl Index':<12} {'Actuator Name':<40} {'Joint Name':<40}")
    print("-" * 100)

    for act_id in range(m.nu):
        act_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id)

        # Get the joint controlled by this actuator
        # Actuators control via transmission
        trans_id = m.actuator_trnid[act_id, 0]
        trans_type = m.actuator_trntype[act_id]

        # Simplified: find the joint name from actuator
        joint_id = m.actuator_trnid[act_id, 0]
        if trans_type == 0:  # mjTRN_JOINT
            joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        else:
            joint_name = "?"

        print(f"{act_id:<12} {act_name:<40} {joint_name:<40}")

    # =========================================================================
    # Sample Data State
    # =========================================================================
    print(f"\n{'='*80}")
    print("SAMPLE DATA STATE (from MjData)")
    print(f"{'='*80}\n")

    print("QPOS (first 7 are typically [x, y, z, qw, qx, qy, qz]):")
    print(f"  Shape: {d.qpos.shape}")
    print(f"  Values:\n{d.qpos}")

    print("\nQVEL (velocities, same structure as qpos dimensions):")
    print(f"  Shape: {d.qvel.shape}")
    print(f"  Values:\n{d.qvel}")

    print("\nCTRL (control inputs for actuators):")
    print(f"  Shape: {d.ctrl.shape}")
    print(f"  Values:\n{d.ctrl}")

    # =========================================================================
    # Body positions and orientations
    # =========================================================================
    print(f"\n{'='*80}")
    print("BODY FRAMES (from MjData)")
    print(f"{'='*80}\n")

    print(f"{'Body Name':<40} {'Position (m)':<30} {'Quat (w,x,y,z)':<30}")
    print("-" * 100)

    for body_id in range(min(m.nbody, 30)):  # Limit to first 30 bodies
        body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id)
        pos = d.xpos[body_id]
        quat = d.xquat[body_id]
        pos_str = f"({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})"
        quat_str = f"({quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f})"
        print(f"{body_name:<40} {pos_str:<30} {quat_str:<30}")

    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect MuJoCo model state mappings")
    parser.add_argument("--config", type=str, default=None,
                      help="YAML config file (reads xml_path from it)")
    parser.add_argument("--xml", type=str, default=None,
                      help="Direct path to MuJoCo XML model file")
    args = parser.parse_args()

    xml_path = None
    config_dir = None
    if args.xml:
        xml_path = args.xml
    elif args.config:
        config = load_config(args.config)
        xml_path = config.get('xml_path')
        config_dir = config['_config_dir']
    else:
        # Try default from deploy config
        default_config = os.path.join(_REPO_ROOT, 'h12_adaptive_policy/deploy/h12_fame.yaml')
        if os.path.exists(default_config):
            config = load_config(default_config)
            xml_path = config.get('xml_path')
            config_dir = config['_config_dir']

    if xml_path is None:
        print("Error: Could not find XML path. Specify via --xml or --config")
        return 1

    # Resolve relative paths
    if not os.path.isabs(xml_path):
        base_dir = config_dir if config_dir is not None else _REPO_ROOT
        xml_path = os.path.normpath(os.path.join(base_dir, xml_path))

    if not os.path.exists(xml_path):
        print(f"Error: XML file not found: {xml_path}")
        return 1

    inspect_model(xml_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
