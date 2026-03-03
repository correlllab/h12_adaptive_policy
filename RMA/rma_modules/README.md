# RMA Modules — File-by-File Explanation

These modules implement **Rapid Motor Adaptation (RMA)** (Kumar et al., RSS 2021) for Isaac Gym. In the paper:

- **Training (sim)**: A privileged **extrinsic vector e_t** (e.g. payload, friction, motor strength) is encoded into a **latent z_t**. The policy is conditioned on **(observation, z_t)**. So the policy learns to use z_t to adapt.
- **Deployment**: e_t is not available. An **adaptation module** takes a **history of (observation, action)** and predicts **ẑ_t**. The policy runs on **(observation, ẑ_t)** and thus adapts online.

Our **e_t** is H12-specific: **15 upper-body joint positions** (read from `dof_pos`) + **5 force values** (sampled; resampled with prob 0.004 per step): **torso ±30 N per axis**, **hands 0–30 N downward each**. These same forces are **applied to the simulation**. All force ranges are defined in `env_factor_spec.py` (single source of truth); sampling and decoder output scaling use them.

---

## 1. `env_factor_spec.py`

**Role:** Defines the **extrinsic vector e_t**: its size, layout, and names. No neural nets.

**RMA paper:** The paper uses an extrinsic vector **e** (or **e_t**) that summarizes “what the robot doesn’t directly observe” but affects dynamics (payload, friction, motor strength, etc.). We define the same concept for H12 + Isaac Gym.

**Contents:**
- **`RmaEtSpec`**  
  Dataclass with:
  - `upper_body_dof_dim=15`, `torso_force_dim=3`, `hand_force_dim=1`, `num_hands=2`  
  → **e_t dim = 20**: 15 joint positions + 3 torso force + 1 left hand + 1 right hand.
  - Properties: `dim`, `upper_body_slice`, `torso_force_slice`, `left_wrist_force_slice`, `right_wrist_force_slice` for indexing e_t.
- **`DEFAULT_ET_SPEC`**  
  Single default spec instance.
- **`UPPER_BODY_JOINT_NAMES`**  
  Tuple of 15 joint names (torso + left arm 7 + right arm 7) so e_t can be filled from `dof_pos` by name.
- **`RMA_FORCE_BODY_NAMES`**  
  Tuple of 3 body names for applying forces: `torso_link`, `left_wrist_roll_link`, `right_wrist_roll_link`.
- **`TORSO_FORCE_RANGE`**, **`HAND_FORCE_RANGE`**  
  Uniform sampling ranges in N: torso ±30 per axis, hands 0–30 downward. Used by sampling (`gym_et_builder.sample_rma_forces`) and decoder output scaling (`env_factor_decoder._register_output_ranges`).

---

## 2. `env_factor_encoder.py`

**Role:** **Encoder** in the RMA sense: maps **e_t → z_t**. Used only in simulation when e_t is available (privileged).

**RMA paper:** The encoder is a neural net **e → z**. The policy is trained with **(obs, z)** so it learns to rely on z. Our encoder is an MLP e_t → z_t, matching that role.

**Contents:**
- **`_build_mlp`**  
  Helper: builds an MLP with configurable layers and activation (e.g. ELU).
- **`EnvFactorEncoderCfg`**  
  - `in_dim=20` (e_t size), `latent_dim=8`, `hidden_dims=(256, 128)`, `activation="elu"`.
- **`EnvFactorEncoder`**  
  - `forward(env_factors)` with shape `(N, 20)` → `(N, latent_dim)`.  
  - No Isaac Lab / Isaac Gym ops inside; pure PyTorch.

**Correctness:** Input dim 20 matches `RmaEtSpec.dim`. Output dim (e.g. 8) is the latent size used by the policy and by the adaptation module; consistent with the paper’s “small latent” design.

---

## 3. `env_factor_decoder.py`

**Role:** **Decoder**: maps **z_t → e_t** (reconstruct extrinsic from latent). Used for an optional **reconstruction loss** during training, not at deployment.

**RMA paper:** An optional decoder **z → e** is trained with reconstruction loss so that z carries enough information about e. We do the same: decoder(z_t) ≈ e_t.

**Contents:**
- **`EnvFactorDecoderCfg`**  
  - `in_dim=8` (latent), `out_dim=20` (e_t), hidden layers, and **output ranges** for scaling:  
    joint range, torso force range, hand force range (so decoded e_t lies in plausible intervals).
- **`EnvFactorDecoder`**  
  - MLP z_t → raw logits, then **sigmoid + linear map** per dimension to `[min, max]` using the registered ranges.  
  - `forward(latent)`, `compute_reconstruction_loss(latent, e_t_target)`, `get_factor_predictions(latent)` (returns dict of upper_body, torso_force, left/right wrist force slices).
- Ranges: 15 × joint range, 3 × torso force range, 2 × hand force range → 20 dims.

**Correctness:** in_dim = encoder’s latent_dim; out_dim = 20 = e_t.dim. Reconstruction loss is MSE(decoder(z_t), e_t), as in the paper. Slices match `RmaEtSpec`.

---

## 4. `adaptation_module.py`

**Role:** **Adaptation module** for deployment: given a **history of (observation, action)**, output **ẑ_t**. No e_t; only proprio/obs + actions.

**RMA paper:** At test time, an adaptation network takes **history of (proprioception, action)** and outputs **ẑ**. Policy then uses **(obs, ẑ)**. The paper uses a 1D CNN over time. We do the same: 1D CNN over a flattened window of (obs, action) per timestep.

**Contents:**
- **`Adaptation1DCNNCfg`**  
  - `in_channels` = obs_dim + action_dim (per timestep), `history_length` (e.g. 50), `latent_dim` (e.g. 8), `hidden_dims` for the CNN.
- **`Adaptation1DCNN`**  
  - Input: **(B, history_length * in_channels)**  
  - Reshape to **(B, in_channels, history_length)** and run 1D convs (kernel 3, padding 1) + ReLU, then flatten and FC → **(B, latent_dim)**.  
  - No e_t, no privileged info; only history of (obs, action).

**Correctness:** Input is “history of (obs, action)”; output dim = encoder’s latent_dim. Matches the paper’s adaptation network role and 1D temporal structure. No dependency on Isaac Lab or Gym inside the module.

---

## 5. `gym_et_builder.py`

**Role:** **Build e_t from Isaac Gym state**. Simulation-only: uses `dof_pos` and `contact_forces` to build the 20D e_t that the encoder (and decoder target) expect.

**RMA paper:** In the paper, e is whatever the sim exposes (e.g. payload, friction). We don’t have a single “e” API in Gym, so we define e_t and this builder: it’s the **concrete implementation** of “e_t from current sim state” for H12.

**Contents:**
- **`build_et_from_gym(dof_pos, contact_forces, body_names, dof_names, et_spec=None, force_scale=1.0)`**  
  - Uses `UPPER_BODY_JOINT_NAMES` and `RMA_FORCE_BODY_NAMES` to index into `dof_pos` and `contact_forces`.  
  - **Torso:** 3D force at `torso_link`: `contact_forces[:, torso_idx, :]`.  
  - **Hands:** 1D downward each: `-contact_forces[:, wrist_idx, 2]` (Z up → downward = −Fz).  
  - Concatenates: [upper_body(15), torso_force(3), left_hand(1), right_hand(1)], optionally scaled by `force_scale`.  
  - Returns **(num_envs, 20)**.

**Correctness:** Output layout and dimension match `RmaEtSpec` and encoder/decoder (20D). Torso 3D vs hands 1D matches the intended semantics (base 3D force, hands load-carrying 1D).

---

## 6. `__init__.py`

**Role:** Package root: documents the RMA flow (training vs deploy) and re-exports the public API.

**Exports:**  
`EnvFactorEncoder`, `EnvFactorDecoder`, `Adaptation1DCNN` and configs; `RmaEtSpec`, `DEFAULT_ET_SPEC`, `UPPER_BODY_JOINT_NAMES`, `RMA_FORCE_BODY_NAMES`, `TORSO_FORCE_RANGE`, `HAND_FORCE_RANGE`; `build_et_from_gym`, `sample_rma_forces`, `make_rma_force_tensor`.

---

## Summary vs RMA paper

| Paper concept        | Our implementation                                      |
|---------------------|----------------------------------------------------------|
| Extrinsic e_t       | 15 from `dof_pos` + 5 sampled (torso ±30 N, hands 0–30 N); same forces applied to sim |
| Encoder e → z      | `EnvFactorEncoder`: e_t (20) → z_t (8)                  |
| Policy input (sim) | (observation, z_t) with z_t = encoder(e_t)              |
| Decoder (optional) | `EnvFactorDecoder`: z_t → e_t, MSE loss                  |
| Adaptation (deploy)| `Adaptation1DCNN`: history(obs, action) → ẑ_t            |
| Policy input (deploy) | (observation, ẑ_t)                                  |

All added files are consistent with the RMA paper and with each other (dims 20 for e_t, 8 for z_t, and history-based adaptation).
