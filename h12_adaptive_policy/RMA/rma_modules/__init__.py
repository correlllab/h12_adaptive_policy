"""Reusable neural modules for RMA (Rapid Motor Adaptation).

Aligned with RMA paper (Kumar et al., RSS 2021): encoder maps privileged e_t -> z_t;
policy sees (obs, z_t) in sim; at deploy, adaptation module predicts z_hat from history.

Phase 1 (training, sim):
- Sample forces: sample_rma_forces(); build e_t: build_et_from_gym(dof_pos, torso, left, right, dof_names).
- Apply forces to sim: make_rma_force_tensor() then gym.apply_rigid_body_force_tensors(sim, forces, None, ENV_SPACE).
- Each step: resample forces for envs with prob RMA_RESAMPLE_PROB (0.004) via resample_rma_forces_for_envs().
- Encoder: e_t -> z_t; policy(obs, z_t). Optional: decoder for reconstruction loss.

Phase 2 (deploy):
- Adaptation1DCNN(history of obs, action) -> z_hat; policy(obs, z_hat).

Exports:
- EnvFactorEncoder, EnvFactorDecoder, Adaptation1DCNN and configs
- RmaEtSpec, DEFAULT_ET_SPEC, UPPER_BODY_JOINT_NAMES, RMA_FORCE_BODY_NAMES
- TORSO_FORCE_RANGE, HAND_FORCE_RANGE, RMA_RESAMPLE_PROB
- build_et_from_gym, sample_rma_forces, resample_rma_forces_for_envs, make_rma_force_tensor
"""

from .env_factor_encoder import EnvFactorEncoder, EnvFactorEncoderCfg
from .env_factor_decoder import EnvFactorDecoder, EnvFactorDecoderCfg
from .adaptation_module import Adaptation1DCNN, Adaptation1DCNNCfg
from .env_factor_spec import (
    DEFAULT_ET_SPEC,
    HAND_FORCE_COMPONENT_RANGE,
    HAND_FORCE_MAGNITUDE_RANGE,
    RmaEtSpec,
    RMA_FORCE_BODY_NAMES,
    RMA_RESAMPLE_PROB,
    UPPER_BODY_JOINT_NAMES,
)
from .gym_et_builder import (
    build_et_from_gym,
    make_rma_force_tensor,
    resample_rma_forces_for_envs,
    sample_rma_forces,
)

__all__ = [
    "EnvFactorEncoder",
    "EnvFactorEncoderCfg",
    "EnvFactorDecoder",
    "EnvFactorDecoderCfg",
    "Adaptation1DCNN",
    "Adaptation1DCNNCfg",
    "RmaEtSpec",
    "DEFAULT_ET_SPEC",
    "UPPER_BODY_JOINT_NAMES",
    "RMA_FORCE_BODY_NAMES",
    "HAND_FORCE_MAGNITUDE_RANGE",
    "HAND_FORCE_COMPONENT_RANGE",
    "RMA_RESAMPLE_PROB",
    "build_et_from_gym",
    "sample_rma_forces",
    "resample_rma_forces_for_envs",
    "make_rma_force_tensor",
]
