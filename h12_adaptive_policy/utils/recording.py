"""MuJoCo video recording: camera, frame overlays, scene markers, mp4 writers.

Pulled out of manip_ik_demo.py so the sim runner only deals with the simulation
loop; everything visualization-related lives here.
"""
import os
import numpy as np
import mujoco


# ─── Camera ─────────────────────────────────────────────────────────────────

def make_manip_camera(lookat=(0.1, 0.0, 0.9), distance=3.2, azimuth=150.0, elevation=-12.0):
    """Default camera for the single-arm manip demo (faces the robot from the right)."""
    cam = mujoco.MjvCamera()
    cam.lookat[:] = list(lookat)
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


# ─── 2-D text overlay on rendered frames ───────────────────────────────────

def overlay_text(frame, lines, x0=10, y0=24, line_h=26, font_scale=0.6, thickness=2):
    """Draw ``lines = [(text, (r,g,b)), ...]`` onto a rendered RGB frame.

    Returns a NEW contiguous array (the input is not mutated). cv2 is imported lazily
    so non-recording paths don't pay for it.
    """
    import cv2
    f = np.ascontiguousarray(frame)
    y = y0
    for txt, col in lines:
        cv2.putText(f, txt, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, col,
                    thickness, cv2.LINE_AA)
        y += line_h
    return f


# ─── 3-D scene markers (force arrows, target spheres, error rods) ──────────

def _scene_connector(scene, gtype, width, a, b, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.zeros(3), np.zeros(3), np.eye(3).reshape(9),
                        np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, gtype, width, np.asarray(a, np.float64), np.asarray(b, np.float64))
    scene.ngeom += 1


def add_force_arrow(scene, origin, force, scale=0.011, rgba=(1.0, 0.2, 0.1, 1.0), width=0.02):
    """Arrow at ``origin`` pointing along ``force``, length = ``|force| * scale``.

    No-op if the force magnitude is effectively zero.
    """
    if float(np.linalg.norm(force)) < 1e-6:
        return
    to = np.asarray(origin, np.float64) + np.asarray(force, np.float64) * scale
    _scene_connector(scene, mujoco.mjtGeom.mjGEOM_ARROW, width, origin, to, rgba)


def add_marker(scene, pos, radius=0.045, rgba=(0.1, 0.95, 0.2, 1.0)):
    """Sphere marker at ``pos`` (e.g., commanded EE target)."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0, 0], np.float64),
                        np.asarray(pos, np.float64),
                        np.eye(3).reshape(9),
                        np.asarray(rgba, np.float32))
    scene.ngeom += 1


def add_error_line(scene, a, b, rgba=(1.0, 0.95, 0.1, 1.0), width=0.007):
    """Capsule rod from ``a`` to ``b`` (e.g., hand-to-target error vector)."""
    if float(np.linalg.norm(np.asarray(a) - np.asarray(b))) < 1e-4:
        return
    _scene_connector(scene, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b, rgba)


# ─── MP4 writers ────────────────────────────────────────────────────────────

def save_mp4(path, frames, fps=25, codec="libx264", quality=8):
    """Write ``frames`` (list/iterable of HxWx3 uint8) to an mp4 at ``path``.

    Creates parent directories if needed. Returns ``path``.
    """
    import imageio
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, codec=codec, quality=quality)
    return path


def sidebyside_frames(left_frames, right_frames, divider_px=4, divider_rgb=(255, 255, 255)):
    """Concatenate two clips horizontally with a thin vertical divider.

    Truncates to the shorter clip. Frame heights must match.
    """
    n = min(len(left_frames), len(right_frames))
    if n == 0:
        return []
    h = left_frames[0].shape[0]
    div = np.full((h, divider_px, 3), 0, np.uint8)
    div[:] = divider_rgb
    return [np.concatenate([left_frames[i], div, right_frames[i]], axis=1) for i in range(n)]


def save_sidebyside_mp4(path, left_frames, right_frames, fps=25, divider_px=4):
    """Convenience: build side-by-side frames and write them to ``path`` as mp4."""
    frames = sidebyside_frames(left_frames, right_frames, divider_px=divider_px)
    return save_mp4(path, frames, fps=fps)
