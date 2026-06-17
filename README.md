# h12_adaptive_policy

Humanoid policy adaptive to end-effector payloads.

## Installation

This project runs in a `conda` environment (Python 3.10).

1. Clone the repo and initialize its submodules (`unitree_sdk2_python` and
   `h12_ros2_controller`):

    ```bash
    git clone https://github.com/correlllab/h12_adaptive_policy.git
    cd h12_adaptive_policy
    git submodule update --init --recursive
    ```

2. Create and activate the environment from `environment.yml`:

    ```bash
    conda env create -f environment.yml
    conda activate adaptive_env
    ```

3. Install the vendored submodule packages (editable) into the env, plus the
   safety layer straight from GitHub:

    ```bash
    pip install -e submodules/unitree_sdk2_python
    pip install -e submodules/h12_ros2_controller
    # h12_safety_layer is not modified here, so install it directly from GitHub.
    # --no-deps: its runtime deps (numpy/scipy/pyyaml/unitree) are already in the env.
    pip install --no-deps git+https://github.com/correlllab/h12_safety_layer.git@main
    ```

   The Unitree SDK must be installed **editable** — a non-editable wheel install
   drops its `b2` subpackage and the import fails.

4. Smoke-test the install:

    ```bash
    python -c "import mujoco, torch, pinocchio, h12_ros2_controller; print('ok')"
    ```

5. Run the Mujoco RMA deployment from the repo root:

    ```bash
    python h12_adaptive_policy/deploy/mujoco_deploy_h12_rma.py
    ```

## Files

- `data/` contains saved data such as model checkpoint and evaluation results.
- `figures` contains generated figures.
- `h12_adaptive_policy/`: contains source code.
    - `deploy/` contains scripts to deploy the policy and run experiments.
    - `example/` contains example scripts.
    - `plot/` contains scripts to plot figures.
    - `RMA/` contains implementation of the RMA modules.
- `submodules/`: contains external dependencies, pulled by `git submodule update --init --recursive`:
    - `unitree_sdk2_python` — Unitree DDS Python SDK.
    - `h12_ros2_controller` — FrameController / `RobotModel` kinematics stack.
- [`h12_safety_layer`](https://github.com/correlllab/h12_safety_layer) — command safety layer
  (joint-limit clipping + e-stop); installed from GitHub (pip), not vendored.

## Usage

- Run scripts from the repo root, e.g. the MuJoCo deploy:
  `python h12_adaptive_policy/deploy/mujoco_deploy_h12_rma.py`.
- The earlier simulation experiments (the 6D hand-force sweeps that produced the
  force-sphere figures) are archived under [`old_experiments/`](old_experiments/);
  new simulation experiments are being built on top of the current runner.

## Safety layer

All motor commands are routed through
[`h12_safety_layer`](https://github.com/correlllab/h12_safety_layer), which clips each command to
the H1-2 joint limits and enforces an e-stop. It is **installed from GitHub** (see Installation),
not vendored. Joint ordering is the shared 27-joint order (legs 0-11, torso 12, left arm 13-19,
right arm 20-26).

### Real robot (`deploy_real/`)

The deploy configs publish to the safety layer's **input** topic `rt/safety/lowcmd_in` instead of
straight to `rt/lowcmd`, so the relay must be running for any command to reach the robot — this is
the intended fail-safe. In a separate terminal, on the **same DDS domain / network interface**, run
the relay (installed as a module):

```bash
python -m h12_safety_layer.script.safety_layer_main --config default_safety_full.yaml
```

It clips + monitors incoming commands and republishes the filtered command to `rt/lowcmd` at 500 Hz.
(If running the module entry point is inconvenient, clone the repo and run/colcon-build it per its
own README.) To bypass it (not recommended), restore `lowcmd_topic: "rt/lowcmd"` in the deploy config.

### Simulation (`deploy/mujoco_deploy_h12_rma.py`)

The MuJoCo deploy reuses the same `h12_safety_layer` joint-position limits to clamp every target
joint position before the PD controller (joints matched by name, so sim and hardware stay
consistent). It is enabled by default; set `safety_clip: false` to bypass, or `safety_config:
<path-to-preset.yaml>` to derive bounds from a safety preset instead of the default joint limits.
