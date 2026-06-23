# h12_adaptive_policy

FAME — a humanoid leg policy adaptive to end-effector payloads, with three real-robot experiments scripted: left-hand pick-and-place, right-hand pick-and-place, and bimanual carry with torso rotation + drop. Two sim entry points: a standalone MuJoCo runner with the full plotting suite, and a DDS-decoupled controller that drops directly onto a real Unitree H1_2 (via `h1_mujoco` for sim or the on-robot DDS for hardware).

## Installation

This project supports both `uv` and `conda` environments on Python 3.10.
`uv` is recommended because it installs the root dependencies, editable
submodules, Git-based safety layer, and video extras from one locked project
configuration.

1. Clone the repo and initialize its submodules (`unitree_sdk2_python` and `h12_ros2_controller`):

    ```bash
    git clone https://github.com/correlllab/h12_adaptive_policy.git
    cd h12_adaptive_policy
    git submodule update --init --recursive
    ```

2. Create the environment with `uv` (recommended):

    ```bash
    uv sync
    ```

    This creates `.venv/` and installs `unitree_sdk2_python` and
    `h12_ros2_controller` as editable path dependencies. Run commands through
    `uv run ...`, which activates the environment automatically.

3. Or create and activate the environment with `conda`:

    ```bash
    conda env create -f environment.yml
    conda activate adaptive_env
    ```

4. For `conda` only, install the vendored submodule packages (editable), plus
   the safety layer and video extra:

    ```bash
    pip install -e submodules/unitree_sdk2_python
    pip install -e submodules/h12_ros2_controller
    pip install --no-deps git+https://github.com/correlllab/h12_safety_layer.git@main
    pip install "imageio[ffmpeg]"   # for video recording (--sweep_video / --video)
    ```

    The Unitree SDK must be installed **editable** — a non-editable wheel install drops its `b2` subpackage and the import fails.

5. Smoke-test the install:

    ```bash
    uv run python -c "import mujoco, torch, pinocchio, h12_ros2_controller, h12_safety_layer, imageio; print('ok')"
    # or, inside the conda env:
    python -c "import mujoco, torch, pinocchio, h12_ros2_controller, h12_safety_layer, imageio; print('ok')"
    ```

## Experiments designed

Three tasks are scripted; all are runnable in sim, with the DDS path ready for real-robot deployment:

| Task | YAML | Notes |
|---|---|---|
| **Right-hand pick-and-place** | `deploy/single_arm_manip.yaml :: right_hand_manip` | 4 EE waypoints, asymmetric load. Headline FAME comparison (clearest pelvis-drift win). |
| **Left-hand pick-and-place** | `deploy/single_arm_manip.yaml :: left_hand_manip` | Mirror of the right task. |
| **Bimanual carry** | `deploy/bi_manual_carry.yaml :: bimanual_carry` | Both hands hold a payload through a torso yaw sweep, then drop. Tests symmetric load + angular-momentum disturbance + drop transient. |

## Run modes (`manip_ik_demo.py` — standalone MuJoCo)

This is the rich sim runner with all the visualization and plotting features. Loads `h1_2_magpie_fame.xml` directly into MuJoCo.

```bash
# Single FAME vs no-FAME comparison (default: right_hand_manip, 1 kg payload)
python h12_adaptive_policy/deploy/manip_ik_demo.py --task right_hand_manip

# Live MuJoCo viewer (real-time, one condition)
python h12_adaptive_policy/deploy/manip_ik_demo.py --task right_hand_manip --view

# Payload sweep with multi-seed mean ± std error bars
python h12_adaptive_policy/deploy/manip_ik_demo.py --task right_hand_manip --sweep --seeds 5

# On-the-fly payload schedule (FAME-vs-no-FAME time series + plot_adapt)
python h12_adaptive_policy/deploy/manip_ik_demo.py --task right_hand_manip --adapt

# Side-by-side payload sweep video (FAME left, no-FAME right; 1-10 kg, force arrows + status ball)
python h12_adaptive_policy/deploy/manip_ik_demo.py --task right_hand_manip --sweep_video

# Bimanual carry
python h12_adaptive_policy/deploy/manip_ik_demo.py --task bimanual_carry
python h12_adaptive_policy/deploy/manip_ik_demo.py --task bimanual_carry --view
python h12_adaptive_policy/deploy/manip_ik_demo.py --task bimanual_carry --sweep --seeds 5
python h12_adaptive_policy/deploy/manip_ik_demo.py --task bimanual_carry --sweep_video    # 3-10 kg
```

Plots are written to `simulation_exp/figures/`, videos to `simulation_exp/videos/`.

## Run modes (`manip_ik_demo_dds.py` — DDS-decoupled)

Same control logic as above, but reads/writes only through DDS topics — no MuJoCo handle. The same binary runs against `h1_mujoco`'s sim server (default) or a real H1_2 by switching the network interface.

```bash
# Terminal A: sim
cd ~/isaac_gym_projects/h1_mujoco
python h12_mujoco.py                  # default scene (handless, with elastic band)
# (or --fixed for pelvis-welded, the cleanest smoke test)

# Terminal B: controller
cd ~/isaac_gym_projects/h12_adaptive_policy
python h12_adaptive_policy/deploy/manip_ik_demo_dds.py --task right_hand_manip
```

After the task summary, the controller keeps the FAME leg policy running indefinitely; Ctrl-C exits with soft damping.

For offline FAME-vs-no-FAME comparison (save trajs from two runs, then compare):

```bash
python h12_adaptive_policy/deploy/manip_ik_demo_dds.py --task right_hand_manip --save_traj /tmp/fame.npz
# restart the sim, then:
python h12_adaptive_policy/deploy/manip_ik_demo_dds.py --task right_hand_manip --no_encode --save_traj /tmp/no_fame.npz
python h12_adaptive_policy/deploy/compare_plot_dds.py --fame /tmp/fame.npz --no_fame /tmp/no_fame.npz --out simulation_exp/figures/compare_dds.png
```

## Files

- `h12_adaptive_policy/` — source code.
    - `deploy/` — entry-point scripts:
        - `manip_ik_demo.py` — standalone MuJoCo runner; full plotting + sweep + video.
        - `manip_ik_demo_dds.py` — DDS-decoupled controller (for `h1_mujoco` sim or real H1_2).
        - `mujoco_deploy_h12_rma.py` — core utilities (`build_et_mujoco`, `compute_observation`, `load_safety_q_clip`, …) imported by every runner. Also runnable standalone for the legacy RMA demo.
        - `compare_plot_dds.py` — offline FAME-vs-no-FAME pelvis-drift comparison from `.npz` trajs.
        - `sweep_rma_ablation.py` — older FAME sweep (static / dynamic / pickplace / envelope).
        - `single_arm_manip.yaml`, `bi_manual_carry.yaml`, `h1_2_rma_arm_magpie_fame.yaml` — task / model configs.
    - `deploy_real/` — real-robot DDS runner (squat policy variant).
    - `utils/` — shared helpers (IK, schedule builders, plotting, recording).
    - `RMA/` — RMA-module sources, including the trained `EnvFactorEncoder`.
    - `example/`, `plot/`, `fame_policy/` — utilities and saved checkpoints.
- `h1_2/` — MuJoCo model + meshes for the H1-2 with Magpie grippers.
- `simulation_exp/` — output directory for plots (`figures/`) and videos (`videos/`).
- `submodules/`:
    - `unitree_sdk2_python` — Unitree DDS Python SDK.
    - `h12_ros2_controller` — Pinocchio `RobotModel` + IK utilities.
- [`h12_safety_layer`](https://github.com/correlllab/h12_safety_layer) — command safety relay (joint-limit clipping + e-stop). Installed from GitHub, not vendored.

## Safety layer

All real-robot motor commands should route through [`h12_safety_layer`](https://github.com/correlllab/h12_safety_layer), which clips each command to the H1-2 joint limits and enforces an e-stop. Joint ordering is the shared 27-joint Unitree HG order (legs 0-11, torso 12, left arm 13-19, right arm 20-26).

### `manip_ik_demo.py` (standalone MuJoCo)

The MuJoCo deploy reuses the same `h12_safety_layer` joint-position limits to clamp every target joint position before the PD controller (joints matched by name, so sim and hardware stay consistent). Enabled by default; set `safety_clip: false` in the deploy YAML to bypass, or `safety_config: <path-to-preset.yaml>` to derive bounds from a safety preset.

### `manip_ik_demo_dds.py` (DDS controller)

The DDS controller publishes commands to a topic name controlled by the `TOPIC_LOWCMD` constant at the top of the file:

- **`"rt/lowcmd"`** (default) — direct path to the sim/robot. **No safety enforcement.** Use only for sim experimentation.
- **`"rt/safety/lowcmd_in"`** — safety-layer input topic. The `h12_safety_layer` relay must be running in a separate terminal; it clips/validates and republishes safe commands to `rt/lowcmd`.

Real-robot deployment **must** use the safety-layer path. In a separate terminal, on the same DDS domain / network interface:

```bash
python -m h12_safety_layer.script.safety_layer_main --config default_safety_full.yaml
```

If you set `TOPIC_LOWCMD = "rt/safety/lowcmd_in"` but forget to launch the safety layer, the controller's commands go to a topic with no subscriber → the sim/robot's 100 ms watchdog kicks in and zeros all motor torques (looks like "the policy isn't running"). Always pair the safety topic with the running relay.

## Force estimator (for real-robot adaptive runs)

The encoder input includes the per-hand force (3D, Newtons, world frame). In sim, this is the **privileged** payload `kg * g` taken from the task YAML. For a real-robot run where the actual payload may differ from the script (or be unknown), a force estimator is needed.

A scaffolding for this is in `manip_ik_demo_dds.py`:

```python
USE_FORCE_ESTIMATOR = False   # top of the file
```

- **`False`** (default): encoder is fed the commanded `kg * g`. Sufficient for sim ablations and scripted real-robot demos.
- **`True`**: the controller will call into the (currently commented-out) inverse-dynamics estimator inside `run_manip_dds`'s `if USE_FORCE_ESTIMATOR:` branch. A reference implementation using `h12_ros2_controller`'s `RobotModel.get_frame_wrench` is included as comments; uncomment + validate before flipping the flag.

A real estimator's contract:
- One 3-vector per hand, **world frame**, **Newtons**.
- Sign: force ON the wrist (e.g., a 2 kg hanging payload → ≈ `[0, 0, -19.6]`).
- Magnitudes > 30 N are direction-preserved clipped inside `build_et_mujoco` — no need to clip at the call site.

Validation note: in the DDS sim path, the controller does not inject `xfrc_applied` (no DDS topic for force injection), so inverse-dynamics estimators in sim will read ≈ 0 regardless of the commanded payload. Validate on hardware, or first add a `rt/sim/xfrc_applied` topic to `h1_mujoco`.

## Quick-start checklist for first real-robot run

1. Pull the latest `h1_2_handless.urdf` into `h1_2/h1_2_handless.urdf`.
2. Set `TOPIC_LOWCMD = "rt/safety/lowcmd_in"` in `manip_ik_demo_dds.py`.
3. Launch `h12_safety_layer.script.safety_layer_main` with conservative limits (e.g., `tight_safety_full.yaml`) and `estop.enabled: true`.
4. Place the robot in the FAME-trained default standing pose; tether/harness for the first attempt.
5. Run the controller with the correct network interface, e.g.:
    ```bash
    python h12_adaptive_policy/deploy/manip_ik_demo_dds.py --task right_hand_manip --payload_kg 1 --net eth0
    ```
6. Start light (≤ 1 kg payload) and watch the safety-layer log for clipped commands.
