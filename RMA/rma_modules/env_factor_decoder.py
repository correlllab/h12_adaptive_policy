"""Environment factor decoder for reconstructing e_t from latent z_t or observations.

This module provides the inverse of the env_factor_encoder: given a latent z_t (or observations),
it reconstructs/predicts the environment factors e_t. Can be trained with supervision from actual
e_t data collected during training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _build_mlp(in_dim: int, hidden_dims: list[int], out_dim: int, activation: str = "elu") -> nn.Sequential:
    """Build a multi-layer perceptron with specified architecture."""
    if activation == "elu":
        act: type[nn.Module] = nn.ELU
    elif activation == "relu":
        act = nn.ReLU
    elif activation == "tanh":
        act = nn.Tanh
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    layers: list[nn.Module] = []
    last_dim = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last_dim, h))
        layers.append(act())
        last_dim = h
    layers.append(nn.Linear(last_dim, out_dim))
    return nn.Sequential(*layers)


@dataclass
class EnvFactorDecoderCfg:
    """Decoder for e_t: 15 upper-body joints (read from sim) + left_hand_xyz(3) + right_hand_xyz(3) = 21.

    Only hand forces are sampled; joints are just read from dof_pos. The joint range here is only
    for scaling the decoder's 15 joint outputs to valid angles when reconstructing e_t (not for sampling).
    """

    in_dim: int = 8
    out_dim: int = 21
    hidden_dims: tuple[int, ...] = (256, 128)
    activation: str = "elu"
    use_output_scaling: bool = True

    # Valid joint angle range (rad) for decoder output dims 0:15 — used only to scale reconstruction to valid e_t; joints are read from sim, not sampled.
    decoder_joint_output_range: tuple[float, float] = (-3.2, 3.2)
    hand_force_component_range: tuple[float, float] = (-30.0, 30.0)  # per axis for 3D hand force


class EnvFactorDecoder(nn.Module):
    """Decodes latent z_t or observations back into environment factors e_t.
    
    Can be trained with MSE loss using actual e_t data as supervision:
        loss = MSE(decoder(z_t), e_t_actual)
    
    This provides a way to:
    1. Debug and understand what the encoder is learning
    2. Reconstruct e_t from latent for analysis
    3. Train a secondary model that predicts e_t from observations (model-based adaptation)
    """

    def __init__(self, cfg: EnvFactorDecoderCfg | None = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else EnvFactorDecoderCfg()
        
        self.net = _build_mlp(
            in_dim=self.cfg.in_dim,
            hidden_dims=list(self.cfg.hidden_dims),
            out_dim=self.cfg.out_dim,
            activation=self.cfg.activation,
        )
        
        if self.cfg.use_output_scaling:
            self._register_output_ranges()

    def _register_output_ranges(self) -> None:
        """Output ranges: 15 joint dims (decoder reconstruction only; joints are read from sim) + 6 hand force components."""
        from .env_factor_spec import DEFAULT_ET_SPEC, HAND_FORCE_COMPONENT_RANGE
        spec = DEFAULT_ET_SPEC
        ranges = []
        for _ in range(spec.upper_body_dof_dim):
            ranges.append(self.cfg.decoder_joint_output_range)
        for _ in range(spec.torso_force_dim):
            pass  # no torso
        for _ in range(spec.num_hands * spec.hand_force_dim):
            ranges.append(self.cfg.hand_force_component_range)
        ranges_tensor = torch.tensor(ranges, dtype=torch.float32)
        self.register_buffer("_output_ranges", ranges_tensor)
    
    def forward(self, latent: torch.Tensor, apply_scaling: bool = True) -> torch.Tensor:
        """Decode latent z_t (or observations) to environment factors e_t.
        
        Args:
            latent: Tensor of shape (N, in_dim), typically z_t from encoder
            apply_scaling: If True, scale outputs to valid e_t ranges
        
        Returns:
            e_t: Tensor of shape (N, out_dim) with decoded environment factors
        """
        if latent.ndim != 2 or latent.shape[-1] != self.cfg.in_dim:
            raise ValueError(
                f"latent must be (N, {self.cfg.in_dim}); got {tuple(latent.shape)}"
            )
        
        # Forward through MLP (raw output, usually unbounded)
        e_t_raw = self.net(latent)
        
        if apply_scaling and self.cfg.use_output_scaling:
            e_t_raw = self._apply_output_scaling(e_t_raw)
        
        return e_t_raw
    
    def _apply_output_scaling(self, e_t_raw: torch.Tensor) -> torch.Tensor:
        """Scale raw network outputs to valid e_t ranges using sigmoid/tanh.
        
        Args:
            e_t_raw: Raw MLP output (N, out_dim)
        
        Returns:
            Scaled e_t (N, out_dim) within valid ranges
        """
        device = e_t_raw.device
        ranges = self._output_ranges.to(device)  # (out_dim, 2)
        
        # Normalize: scale using sigmoid to [0, 1], then map to [min, max]
        e_t_normalized = torch.sigmoid(e_t_raw)  # (N, out_dim) in [0, 1]
        
        min_vals = ranges[:, 0]  # (out_dim,)
        max_vals = ranges[:, 1]  # (out_dim,)
        
        # Scale from [0, 1] to [min, max]
        e_t_scaled = min_vals + e_t_normalized * (max_vals - min_vals)
        
        return e_t_scaled
    
    def compute_reconstruction_loss(
        self,
        latent: torch.Tensor,
        e_t_target: torch.Tensor,
        loss_fn: nn.Module | None = None,
        apply_scaling: bool = True,
    ) -> torch.Tensor:
        """Compute supervised reconstruction loss.
        
        Args:
            latent: Latent encoding (N, in_dim)
            e_t_target: Ground-truth environment factors (N, out_dim)
            loss_fn: Loss function (default: MSE)
            apply_scaling: Whether to apply output scaling
        
        Returns:
            Scalar loss value
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        
        e_t_pred = self.forward(latent, apply_scaling=apply_scaling)
        loss = loss_fn(e_t_pred, e_t_target)
        return loss
    
    def get_factor_predictions(self, latent: torch.Tensor, apply_scaling: bool = True) -> dict[str, torch.Tensor]:
        """Decode and return individual environment factors (e_t layout: 15 joints + left_xyz(3) + right_xyz(3))."""
        from .env_factor_spec import DEFAULT_ET_SPEC
        spec = DEFAULT_ET_SPEC
        e_t = self.forward(latent, apply_scaling=apply_scaling)
        out = {
            "upper_body_joint_pos": e_t[:, spec.upper_body_slice],
            "left_wrist_roll_force": e_t[:, spec.left_wrist_force_slice],
            "right_wrist_roll_force": e_t[:, spec.right_wrist_force_slice],
        }
        if spec.torso_force_dim > 0:
            out["torso_force"] = e_t[:, spec.torso_force_slice]
        return out
