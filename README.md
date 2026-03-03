# h12_adaptive_policy

Humanoid policy adaptive to end-effector payloads.

## Installation

- Initialize submodules via:

    ```bash
    git submodule update --init --recursive
    ```

- Clone the Unitree Python SDK from this [repo](https://github.com/unitreerobotics/unitree_sdk2_python).
- Install python dependencies from `environment.yml` using [`conda`](https://github.com/conda-forge/miniforge):

    ```bash
    conda env create -f environment.yml
    conda activate sim_env
    cd PATH_TO_UNITREE_SDK
    pip install -e .
    ```

## Files

- `data/` contains saved data such as model checkpoint and evaluation results.
- `figures` contains generated figures.
- `h12_adaptive_policy/`: contains source code.
    - `deploy/` contains scripts to deploy the policy and run experiments.
    - `example/` contains example scripts.
    - `plot/` contains scripts to plot figures.
    - `RMA/` contains implementation of the RMA modules.
- `submodules/`: contains external dependencies.

## Usage

- Run scripts from the root directory `python h12_adaptive_policy/deploy/eval_rma_hand_sweep_6d.py`.
