# h12_adaptive_policy

Humanoid policy adaptive to end-effector payloads.

## Installation

Both `uv` and `conda` flows are supported. `uv` is the recommended path: it
resolves the full stack (RMA policy deps + the `h12_ros2_controller` and
`unitree_sdk2_python` submodules) from a single `pyproject.toml` and creates a
local `.venv`.

### Option A: uv (recommended)

1. Install [`uv`](https://docs.astral.sh/uv/) (e.g. `curl -LsSf https://astral.sh/uv/install.sh | sh`).
2. Clone this repo and initialize its submodules:

    ```bash
    git clone https://github.com/correlllab/h12_adaptive_policy.git
    cd h12_adaptive_policy
    git submodule update --init --recursive
    ```

3. Let `uv` create the environment and install every dependency (both the
   `h12_ros2_controller` and `unitree_sdk2_python` submodules are installed as
   editable path dependencies):

    ```bash
    uv sync
    ```

   This creates `.venv/` pinned to Python 3.10. All subsequent commands should
   be invoked via `uv run ...`, which activates the venv automatically.
4. Smoke-test the install:

    ```bash
    uv run python -c "import mujoco, torch, pinocchio, h12_ros2_controller; print('ok')"
    ```

5. Run the Mujoco RMA deployment from the repo root:

    ```bash
    uv run python h12_adaptive_policy/deploy/mujoco_deploy_h12_rma.py
    ```

See [docs/mujoco_frame_controller_howto.md](docs/mujoco_frame_controller_howto.md)
for running the Mujoco sim together with the `h12_ros2_controller` FrameController
stack.

### Option B: conda (legacy)

- Initialize submodules:

    ```bash
    git submodule update --init --recursive
    ```

- The Unitree Python SDK is vendored as a submodule under
  `submodules/unitree_sdk2_python` (cloned by the `git submodule update` above).
- Install python dependencies from `environment.yml`:

    ```bash
    conda env create -f environment.yml
    conda activate adaptive_env
    pip install -e submodules/unitree_sdk2_python
    pip install -e submodules/h12_ros2_controller
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
    - `h12_safety_layer` — command safety layer (joint-limit clipping + e-stop), branch `main`.

## Usage

- Run scripts from the root directory `python h12_adaptive_policy/deploy/eval_rma_hand_sweep_6d.py`.

## Safety layer

All motor commands are routed through
[`h12_safety_layer`](https://github.com/correlllab/h12_safety_layer) (vendored at
`submodules/h12_safety_layer`, branch `main`), which clips each command to the H1-2 joint limits
and enforces an e-stop. Joint ordering is the shared 27-joint order (legs 0-11, torso 12,
left arm 13-19, right arm 20-26).

### Real robot (`deploy_real/`)

The deploy configs publish to the safety layer's **input** topic `rt/safety/lowcmd_in` instead of
straight to `rt/lowcmd`, so the relay must be running for any command to reach the robot — this is
the intended fail-safe. In a separate terminal, on the **same DDS domain / network interface**, run
the relay from the submodule:

```bash
cd submodules/h12_safety_layer
uv run h12_safety_layer/script/safety_layer_main.py --config default_safety_full.yaml
```

It clips + monitors incoming commands and republishes the filtered command to `rt/lowcmd` at 500 Hz.
To bypass it (not recommended), restore `lowcmd_topic: "rt/lowcmd"` in the deploy config.

### Simulation (`deploy/mujoco_deploy_h12_rma.py`)

The MuJoCo deploy reuses the same `h12_safety_layer` joint-position limits to clamp every target
joint position before the PD controller (joints matched by name, so sim and hardware stay
consistent). It is enabled by default; set `safety_clip: false` (or point `safety_config:` at
another preset under `submodules/h12_safety_layer/config/`) in the deploy YAML to change it. The
submodule is added to `sys.path` automatically, so no extra install is needed in the sim env.
