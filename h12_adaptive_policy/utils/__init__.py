"""Shared helpers for the manip / sim-deploy scripts.

Public API (import from ``h12_adaptive_policy.utils`` or ``utils`` when the package
dir is on sys.path):

  math:        quat2R, smoothstep
  IK:          ArmIK, solve_arm_waypoints, build_xyz_schedule, build_payload_profile,
               ARM_JOINTS, ARM_SLICE
  recording:   make_manip_camera, overlay_text, add_force_arrow, add_marker,
               add_error_line, save_mp4, save_sidebyside_mp4, sidebyside_frames
  plotting:    plot_traj, plot_summary, plot_adapt
"""
from .math_helpers import quat2R, smoothstep
from .ik_planning import (
    ArmIK, solve_arm_waypoints, build_xyz_schedule, build_payload_profile,
    ARM_JOINTS, ARM_SLICE,
)
from .recording import (
    make_manip_camera, overlay_text,
    add_force_arrow, add_marker, add_error_line,
    save_mp4, save_sidebyside_mp4, sidebyside_frames,
)
from .plotting import (
    plot_traj, plot_summary, plot_summary_multiseed, plot_adapt, plot_adapt_single,
    plot_carry_compare, plot_carry_summary_multiseed,
)

__all__ = [
    "quat2R", "smoothstep",
    "ArmIK", "solve_arm_waypoints", "build_xyz_schedule", "build_payload_profile",
    "ARM_JOINTS", "ARM_SLICE",
    "make_manip_camera", "overlay_text",
    "add_force_arrow", "add_marker", "add_error_line",
    "save_mp4", "save_sidebyside_mp4", "sidebyside_frames",
    "plot_traj", "plot_summary", "plot_summary_multiseed", "plot_adapt", "plot_adapt_single",
    "plot_carry_compare", "plot_carry_summary_multiseed",
]
