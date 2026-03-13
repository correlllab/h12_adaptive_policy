# Investigation: re3_real_encode_tr1 Disaster Episode

**Date of investigation:** 2026-03-13
**Episode data:** `data/real/re3_real_encode_tr1`
**Duration:** 77 steps, ~1.513 s (20 ms control loop, 50 Hz)
**Wall-clock time:** 22.465 → 23.978 s

---

## Summary

A balancing RL policy trained in simulation was deployed on the H1.2 robot. The policy conditions on an estimated external wrist wrench computed by Pinocchio forward dynamics from logged motor torques. Within **0.16 s** of policy start, Pinocchio computed a left wrist force of **−532 N (Fz)** despite zero real external load. The policy responded by commanding extreme leg torques (up to **428 N·m** at hip pitch), causing the robot to violently kick its legs.

The root cause is **not** abnormal arm torques. It has two compounding factors:

1. **Jacobian geometry:** The `arm_waist_target` configuration places the arms in a raised/bent pose (L: −17°/86° elbow, R: −48° shoulder pitch). This puts the wrist far from the torso rotation axis, creating a **15.6× torso→wrist-Fz amplification** in `pinv(J^T)`.
2. **Torso residual:** Normal structural dynamics from the first leg commands drive a torso `tau_est` of −34 N·m within 0.16 s; the static gravity compensation assigns ~0 N·m to the torso, leaving the full residual unexplained.

The product: 15.6 × 34 N·m = **530 N** apparent wrist force.

This is confirmed by comparison with RE2 (a successful 27-second run of the same policy), where the arms hang at ~0° and the torso→Fz gain is only **1.5×**. RE2 never exceeds 30 N left wrist force across its entire run despite similar torso and leg torque magnitudes.

---

## Data Components

| File | Shape | Contents |
|------|-------|----------|
| `tau.npy` | (77, 27) | Motor estimated torques `tau_est` (N·m), all 27 body joints in `BODY_JOINTS` order |
| `qpos.npy` | (77, 27) | Joint positions (rad) |
| `dq.npy` | (77, 27) | Joint velocities (rad/s) |
| `target_dof.npy` | (77, 27) | Full target joint positions sent to motors (rad) |
| `time.npy` | (77,) | Episode-relative timestamps (s) from `time.time() - start_time` |
| `knee_ankle_tau.csv` | 77 rows | Subset: time + 6 knee/ankle `tau_est` values (N·m); columns: `L_knee, L_ankle_pitch, L_ankle_roll, R_knee, R_ankle_pitch, R_ankle_roll` |

Joint index ordering in all 27-column arrays follows `BODY_JOINTS` in `joint_definition.py`:
- Indices 0–11: legs (l_hip_yaw → r_ankle_roll)
- Index 12: torso
- Indices 13–19: left arm (L_shoulder_pitch → L_wrist_yaw)
- Indices 20–26: right arm (R_shoulder_pitch → R_wrist_yaw)

---

## Timeline

| Step | Time (s) | Δt from start | Event |
|------|----------|----------------|-------|
| 0 | 22.465 | 0.000 | RL policy begins; arm torques small; leg torques near-equilibrium |
| 2 | 22.490 | 0.025 | First sign of leg residual spike (‖τ_legs_residual‖ = 157 N·m); wrench still ~9 N |
| **8** | **22.609** | **0.144** | **Left wrist Fz = −532 N (Pinocchio)** — matches hardcoded comment in `deploy_real.py` |
| 9–15 | 22.63–22.75 | 0.16–0.29 | Left wrist force escalates to 715 → 1584 N as leg torques compound |
| 43–46 | 23.31–23.38 | 0.85–0.91 | Peak left wrist force: **2596 N** |
| 64 | 23.737 | 1.27 | Peak lower-body torque: l_hip_pitch = **428 N·m** |
| 76 | 23.978 | 1.51 | Episode end (robot stopped) |

---

## Arm Torque Analysis

The arm joint torques at the onset are **not abnormal**:

| Joint | First step τ (N·m) | Max |τ| (N·m) | Expected gravity comp |
|-------|--------------------|---------------------|------------------------|
| L_shoulder_pitch | −2.11 | 46.1 | ~4–8 Nm at −17° |
| L_elbow | −0.97 | 10.9 | ~1 Nm at 87° |
| R_shoulder_pitch | −12.57 | 83.3 | ~10–14 Nm at −48° |
| R_shoulder_roll | −7.12 | 40.9 | ~4–5 Nm at −27° |
| R_elbow | −2.99 | 37.4 | ~2–3 Nm at 30° |

Arm targets at episode start closely match actual positions (e.g., L_shoulder_pitch: −0.29 rad measured vs −0.30 rad target; R_shoulder_pitch: −0.84 vs −0.90 rad). The arm residual norm (τ_arm − τ_grav_arm) at step 8 is only **1.55 N·m** — consistent with correct gravity compensation.

**Conclusion: arm torques are NOT the direct cause of the 532 N wrench.**

---

## RE2 vs RE3 Comparison

The critical evidence comes from comparing RE3 with RE2 (`data/real/re2_real_encode`), a successful 27-second run of the same policy.

### Arm configuration at episode start

| Joint | RE3 (disaster) | RE2 (good) |
|-------|---------------|------------|
| L_shoulder_pitch | **−16.6°** | 0.5° |
| L_elbow | **86.2°** | 1.7° |
| R_shoulder_pitch | **−47.9°** | 0.8° |
| R_shoulder_roll | **−26.7°** | 0.1° |

RE2's arms are nearly straight down (natural hanging). RE3 starts with arms in the `arm_waist_target` pose from the config (`−0.3 / −0.9` rad shoulder pitch, `1.5 / 0.5` rad elbow).

### Resulting Jacobian properties (left wrist, step 0)

| | Torso → left wrist Fz gain | Jacobian cond. |
|-|---------------------------|----------------|
| **RE3** | **15.6 N per N·m** | 39.9 |
| **RE2** | **1.5 N per N·m** | 12.9 |

### Side-by-side first 15 steps

```
         torso_τ  torso_res |  L_wrist Fz   L|F|  |  l_knee_τ  r_knee_τ
RE3 s=0     0.12       0.12 |      -15.2   15.3  |     -7.5      -9.5
RE3 s=5    14.24      14.24 |      177.8  189.7  |    -62.4     -33.6
RE3 s=8   -34.45     -34.45 |     -532.9  558.2  |    215.1     113.2

RE2 s=0     0.82       0.82 |        6.2    9.4  |     -8.9      -1.9
RE2 s=5     5.98       5.98 |        7.4   21.3  |    -10.0       3.7
RE2 s=8    10.31      10.31 |       16.5   33.2  |    -13.6       5.2
```

RE2 runs for 1357 steps (27 s). Its max left wrist force across the entire run is **29.7 N**. Its max torso torque is **10.5 N·m**. These are similar torso residuals to RE3's first few steps, but the 10× lower Jacobian gain keeps the wrist force completely benign.

---

## Root Cause: Torso Joint Amplification

The Pinocchio wrench estimator in `robot_model.py::get_frame_wrench`:

```python
tau_gravity = self.get_gravity_compensation(q, imu_quat)   # RNEA over all 27 joints
jac = self.get_frame_jacobian(frame_name, q, imu_quat)     # J: 6×27 for wrist frame
wrench = np.linalg.pinv(jac.T) @ (tau - tau_gravity)      # 6-vector: [F; M]
```

This assumes the **only** source of torque residual is the external wrench at the specified frame. In practice, any other joint's unmodelled torque (inertial, contact, dynamic loading) is projected onto the wrist force.

### Per-joint wrench decomposition at step 8 (left wrist)

| Joint | τ_meas (N·m) | τ_grav (N·m) | residual (N·m) | → Fz_contribution (N) |
|-------|-------------|-------------|---------------|-----------------------|
| **torso** | **−34.45** | **~0** | **−34.45** | **−538 N** ← ROOT CAUSE |
| L_sh_pitch | −5.10 | −3.79 | −1.31 | +9 N |
| l_knee | +215.06 | +2.15 | +212.91 | ~0 (leg Jacobian columns ≈ zero for wrist) |
| *(all others)* | — | — | small | < ±5 N each |
| **TOTAL** | | | | **−533 N** |

The **torso joint** sits kinematically between the pelvis and both arms. Its Jacobian column for the wrist frame has a large lever-arm component (~torso-to-wrist distance). This amplifies the torso residual by **~16×** to produce the apparent wrist force.

The Jacobian condition number at step 8 is **39.9** (singular values: 1.86, 1.68, 1.46, 0.39, 0.37, 0.047). The smallest singular value corresponds to a direction amplified by **21×** in the pseudo-inverse.

### Why the torso residual spikes at step 8

The torso tau_grav is near zero (the arm CoMs are approximately over the torso axis), so τ_grav_torso ≈ 0. But the `tau_est` for the torso at step 8 is −34.45 N·m.

The torso motor is PD-controlled to 0 rad (from `arm_waist_target[0] = 0.0` in the config). The wild leg movements beginning at steps 2–5 create inertial/reaction forces transmitted through the robot structure. The torso motor fights these disturbances, producing a non-zero `tau_est`. Because gravity compensation does not account for this dynamic loading (only static gravity is modelled), the residual appears as an unexplained external wrench at the wrist.

### Cascade

```
RL policy engaged
    → commands large leg position targets from step 1
    → knee/hip torques ramp up (l_knee: 215 N·m by step 8)
    → structural dynamics excite the torso joint
    → torso tau_est = −34 N·m (static gravity comp gives ~0)
    → Pinocchio wrench: pinv(J_wrist^T) @ residual → Fz = −532 N
    → encoder receives e_t = [upper_pos, left_force=−532N_z, ...]
    → RL policy interprets as large leftward/downward push at wrist
    → commands even larger leg torques to "counteract"
    → further excites torso → loop escalates to 2600 N by step 46
```

Note: in the `re3` deploy script, the Pinocchio call **is commented out** (`left_force = np.zeros(3)`), and the hardcoded values in the comments:

```python
# left_force = np.array([-115.08508926, -30.89641282, 532.00016259])
# right_force = np.array([74.82097678, 34.1630265, -37.48545911])
```

were captured from an inspection run on this or a closely related episode. The values match step 8 exactly (−532.9 N Fz confirmed by offline reconstruction). The disaster recorded in `re3_real_encode_tr1` may therefore correspond to the episode where these forces were active, with the deploy script subsequently patched to zero the forces.

---

## Joint Torque Magnitude Summary

All `tau.npy` values are `tau_est` readings — the motor's internal torque estimate. These reflect what the hardware actually delivered, not purely the PD command feedforward.

### Lower body (legs)

| Joint | Mean (N·m) | Max |τ| (N·m) | Peak step |
|-------|-----------|------------------|-----------|
| l_hip_yaw | 8.4 | 90 | 70 |
| l_hip_pitch | **57.3** | **428** | **64** |
| l_hip_roll | 20.5 | 335 | 29 |
| l_knee | 19.1 | 329 | 43 |
| l_ankle_pitch | 5.8 | 41 | 49 |
| l_ankle_roll | 0.9 | 18 | 20 |
| r_hip_yaw | 19.2 | 111 | 70 |
| r_hip_pitch | −50.7 | **397** | 44 |
| r_hip_roll | 8.3 | 357 | 61 |
| r_knee | −7.6 | 333 | 45 |
| r_ankle_pitch | 2.1 | 47 | 71 |
| r_ankle_roll | −1.9 | 15 | 6 |

Hip-pitch and hip-roll motors reach 300–400 N·m — far beyond gravity compensation values (< 30 N·m when standing) and consistent with the robot attempting to apply large ground forces to counteract the imaginary wrist load.

### Upper body

Arm joints stay within moderate bounds (< 84 N·m) even as the lower body goes chaotic. The torso reaches up to **142 N·m** at step 12, consistent with fighting structural dynamics.

---

## Position Tracking

At episode start, the lower body immediately shows tracking errors of 0.35–0.60 rad (20–34°) at hip roll joints. This suggests the robot was not in the policy's expected initial configuration, possibly contributing to large initial leg commands and the rapid escalation.

Arm positions match their targets closely throughout (deviation < 5° on most joints), confirming the arm controller was functioning normally.

---

## Possible Contributing Factors

1. **Torso residual amplification (confirmed, validated against RE2):** Any torso torque not explained by static gravity compensation is amplified ~16× in RE3's arm configuration. In RE2 (arms at 0°) the same mechanism gives only 1.5× — 10× safer. The amplification factor is set entirely by the arm pose at the time of wrench estimation.

2. **No IMU in wrench computation:** `get_frame_wrench` defaults to identity base orientation when no `imu_quat` is passed. With the robot dynamically moving, the base orientation error adds to gravity compensation error, creating additional residuals across all joints.

3. **Full-body pseudo-inverse:** The Jacobian pseudo-inverse `pinv(J^T)` is 6×27. Even joints with nominally zero Jacobian columns (legs) contribute due to numerical imprecision. The torso, being kinematically upstream of the wrist, has large non-zero Jacobian columns.

4. **Initial configuration mismatch:** Hip roll tracking errors up to 0.6 rad at step 0 indicate the robot started in an off-nominal pose, triggering large immediate leg commands.

5. **No wrench sanity clamp:** The wrench estimate is passed raw to the encoder. A simple magnitude clamp (e.g., |F| < 50 N) would have prevented the 532 N value from reaching the policy.

---

## Recommendations

| Priority | Fix |
|----------|-----|
| **Critical** | Add an output clamp to `get_frame_wrench`: `wrench[:3] = np.clip(wrench[:3], -F_max, F_max)` with `F_max` ≈ 50 N |
| **Critical** | Always pass `imu_quat` to `get_frame_wrench` so gravity compensation uses the actual base orientation |
| **High** | Replace full-body pseudo-inverse with an arm-only wrench estimator: restrict τ and J to arm+torso columns only (indices 12–26), reducing contamination from leg dynamics |
| **High** | Verify robot is in the correct initial configuration before engaging the RL policy (check hip roll deviation < 0.1 rad) |
| **Medium** | Log the raw Pinocchio wrench alongside other state in future episodes for real-time monitoring |
| **Medium** | Add a watchdog: if `|wrist_force| > 100 N` disengage policy and enter damping mode |

---

## Reproducibility

To reproduce this analysis:

```bash
# Full text analysis (no pinocchio required)
python scripts/investigate_re3.py --data data/real/re3_real_encode_tr1

# With wrist wrench reconstruction (requires pinocchio venv)
/home/humanoid/Programs/h12_ros2_controller/.venv/bin/python \
    scripts/investigate_re3.py --data data/real/re3_real_encode_tr1 --pinocchio

# Save all plots
/home/humanoid/Programs/h12_ros2_controller/.venv/bin/python \
    scripts/investigate_re3.py --data data/real/re3_real_encode_tr1 \
    --pinocchio --plot --save-plots docs/re3_plots

# Interactive exploration (one step, specific joints)
python scripts/explore_episode.py --data data/real/re3_real_encode_tr1 \
    --step 8 --joints torso l_knee r_knee l_hip_pitch

# With pinocchio at step 8
/home/humanoid/Programs/h12_ros2_controller/.venv/bin/python \
    scripts/explore_episode.py --data data/real/re3_real_encode_tr1 \
    --step 8 --pin
```
