# old_experiments — archived simulation experiments

Snapshot of the **pre-FAME / pre-EE-tracking** simulation experiments, kept for
reference while we formulate the new experiment suite. Real-robot code, the
MuJoCo runner (`h12_adaptive_policy/deploy/mujoco_deploy_h12_rma.py`), the policy
(`h12_adaptive_policy/fame_policy/`), the RMA modules, and the H1-2 model stay in
the main tree — only the old experiment definitions and their outputs are archived
here.

## Contents

- `deploy/` — the old experiment scripts:
  - `eval_rma_hand_sweep_6d_oracle.py` — 6D hand-force sweep with privileged
    (oracle) forces.
  - `eval_rma_hand_sweep_6d_pin.py` — same sweep with Pinocchio force estimation.
  These produced the **force-sphere scatter** (the figure the new
  end-effector-error / load-envelope experiments are meant to replace).
- `data/` — recorded sweep outputs: `oracle/`, `pin/`, `base_eval/` (CSV/PNG).
- `figure/` — generated figures: `oracle/`, `pin/`.

## Running (caveats)

These are frozen as reference, not wired to run from this location. To re-run, note:

- They import the RMA modules from `h12_adaptive_policy/` and expect a deploy
  config — pass `--config h12_adaptive_policy/deploy/h1_2_rma_arm_magpie_fame.yaml`
  (the config was intentionally left in the main tree).
- Their output paths are computed relative to the repo root, so a re-run writes to
  top-level `data/`/`figure/`, not back into this folder.

## New experiments (in progress)

The replacement suite reframes the metric from "did it fall" to **world-frame
end-effector error vs. payload** (Pick / Place / Carry-and-place, FAME ablations).
See the resubmission tracker for the full plan.
