"""Arm IK + waypoint/payload schedules for single-arm manipulation demos.

Uses Pinocchio + Pink on a reduced (arm-only) model: the rest of the H1_2 joints
are locked at a nominal pose (arm_down for the passive arm, torso at the trained
offset). Used by manip_ik_demo.py to:
  - solve offline IK for a list of waypoints (solve_arm_waypoints)
  - re-solve velocity IK every control tick to track a base-frame target (ArmIK)
  - synthesize smooth schedules (build_xyz_schedule, build_payload_profile)
"""
import numpy as np
import pinocchio as pin
import pink
import qpsolvers

from .math_helpers import smoothstep


_ARM_JN = (
    "shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint", "elbow_joint",
    "wrist_roll_joint", "wrist_pitch_joint", "wrist_yaw_joint",
)
ARM_JOINTS = {s: [f"{s}_{j}" for j in _ARM_JN] for s in ("left", "right")}
# slices into the 15-dof upper vector [torso, L_arm(7), R_arm(7)]
ARM_SLICE = {"left": slice(1, 8), "right": slice(8, 15)}

_QP_SOLVERS = ("daqp", "quadprog", "osqp", "proxqp")


def _pick_solver():
    return next(s for s in _QP_SOLVERS if s in qpsolvers.available_solvers)


def _seed_reduced_model(model_body, side, arm_down, torso):
    """Build a Pinocchio reduced model with only the manip arm free.

    All other joints (passive arm, torso, legs) are locked at `arm_down` / `torso`,
    so IK only solves the 7 manip-arm joints.
    """
    qn = pin.neutral(model_body)
    for s in ("left", "right"):
        for jname, val in zip(ARM_JOINTS[s], arm_down):
            qn[model_body.joints[model_body.getJointId(jname)].idx_q] = val
    qn[model_body.joints[model_body.getJointId("torso_joint")].idx_q] = torso
    lock = [model_body.getJointId(n) for n in model_body.names[1:] if n not in ARM_JOINTS[side]]
    return pin.buildReducedModel(model_body, lock, qn)


def solve_arm_waypoints(rm, side, arm_down, torso, waypoints, orientation_cost=0.0,
                        position_cost=50.0, posture_cost=1e-3, max_iters=800, tol_m=1e-3,
                        dt=0.05):
    """Offline IK: arm-only joint targets for each EE waypoint.

    Returns ``(sols, resid)`` where ``sols`` is ``(N, 7)`` joint configs and ``resid`` is
    the per-waypoint position residual (m). Waypoints with residual > 1cm are unreachable.
    """
    rmod = _seed_reduced_model(rm.model_body, side, arm_down, torso)
    rdat = rmod.createData()
    cfg = pink.Configuration(rmod, rdat, np.asarray(arm_down, dtype=float))
    frame = f"{side}_wrist_yaw_link"
    ft = pink.tasks.FrameTask(frame, position_cost=position_cost,
                              orientation_cost=orientation_cost, lm_damping=1.0)
    pt = pink.tasks.PostureTask(cost=posture_cost)
    solver = _pick_solver()
    sols, resid = [], []
    for wp in waypoints:
        rpy = np.deg2rad(wp.get("rpy", [0, 0, 0]))
        R = pin.rpy.rpyToMatrix(*rpy) if orientation_cost > 0 else np.eye(3)
        ft.set_target(pin.SE3(R, np.asarray(wp["xyz"], float)))
        pt.set_target(cfg.q)
        for _ in range(max_iters):
            cfg.integrate_inplace(pink.solve_ik(cfg, [ft, pt], dt=dt, solver=solver), dt)
            if np.linalg.norm(ft.compute_error(cfg)[:3]) < tol_m:
                break
        sols.append(cfg.q.copy())
        resid.append(float(np.linalg.norm(ft.compute_error(cfg)[:3])))
    return np.array(sols), np.array(resid)


class ArmIK:
    """Closed-loop velocity IK for the manip arm on a reduced (arm-only) model.

    ``step(base_target_xyz, dt)`` drives the wrist frame toward a target given in the
    CURRENT base frame and returns the 7 arm joint angles. Re-solved every control tick
    so the arm actively compensates base motion to keep the hand on its world target.
    """

    def __init__(self, rm, side, arm_down, torso,
                 position_cost=50.0, orientation_cost=0.0, posture_cost=1e-3):
        self.model = _seed_reduced_model(rm.model_body, side, arm_down, torso)
        self.data = self.model.createData()
        self.frame = f"{side}_wrist_yaw_link"
        self.cfg = pink.Configuration(self.model, self.data, np.asarray(arm_down, float))
        self.ft = pink.tasks.FrameTask(self.frame, position_cost=position_cost,
                                       orientation_cost=orientation_cost, lm_damping=1.0)
        self.pt = pink.tasks.PostureTask(cost=posture_cost)
        self.solver = _pick_solver()

    def step(self, base_target, dt):
        self.ft.set_target(pin.SE3(np.eye(3), np.asarray(base_target, float)))
        self.pt.set_target(self.cfg.q)
        vel = pink.solve_ik(self.cfg, [self.ft, self.pt], dt=dt, solver=self.solver)
        self.cfg.integrate_inplace(vel, dt)
        return self.cfg.q.copy()


def _smooth_keyframe_interp(times, values):
    """Closure that smooth-step interpolates ``values`` at the given keyframe ``times``.

    Clamps to endpoint values outside [times[0], times[-1]]. ``values`` may be 1-D
    (scalar series) or 2-D (vector series).
    """
    times = np.asarray(times)
    V = np.asarray(values)

    def at(t):
        if t <= times[0]:
            return V[0]
        if t >= times[-1]:
            return V[-1]
        i = int(np.searchsorted(times, t) - 1)
        span = times[i + 1] - times[i]
        u = smoothstep((t - times[i]) / span) if span > 1e-9 else 1.0
        return V[i] + u * (V[i + 1] - V[i])

    return at


def build_xyz_schedule(waypoints, settle_s, seg_s, hold_s):
    """Base-frame EE target path vs time.

    Hold at the first waypoint through ``settle_s``, then move waypoint-to-waypoint with
    ``seg_s`` moves and ``hold_s`` holds between them. Returns ``(xyz_at, total_s)``.
    """
    pts = [np.asarray(w["xyz"], float) for w in waypoints]
    keys = [(0.0, pts[0]), (settle_s, pts[0])]
    t = settle_s
    for p in pts:
        t += seg_s; keys.append((t, p))
        t += hold_s; keys.append((t, p))
    total = t + 0.5
    times = np.array([k[0] for k in keys])
    P = np.stack([k[1] for k in keys])
    return _smooth_keyframe_interp(times, P), total


def build_payload_profile(settle_s, step_dur_s=0.15):
    """On-the-fly payload (kg) vs time with sudden step changes in the trained 0-3kg range.

    Returns ``(payload_at, step_times)``. ``step_times`` are the instants of sudden change
    (used to draw vertical guides on adaptation plots). ``step_dur_s`` is the short ramp
    duration of each step (shorter = more sudden).
    """
    r = step_dur_s
    keys = [
        (0.0, 0.0),
        (settle_s, 0.0),
        (settle_s + 0.3, 1.5),                                   # initial grasp ~15N
        (settle_s + 2.3, 1.5), (settle_s + 2.3 + r, 3.0),        # +1.5kg -> ~30N (trained edge)
        (settle_s + 4.5, 3.0), (settle_s + 4.5 + r, 0.8),        # drop to ~8N (object removed)
        (settle_s + 6.5, 0.8), (settle_s + 6.5 + r, 2.5),        # +1.7kg -> ~25N
    ]
    steps = [settle_s + 0.3, settle_s + 2.3, settle_s + 4.5, settle_s + 6.5]
    times = np.array([k[0] for k in keys])
    V = np.array([k[1] for k in keys])
    return _smooth_keyframe_interp(times, V), steps
