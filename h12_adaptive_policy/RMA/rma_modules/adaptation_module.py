"""1D CNN adaptation module for RMA phase-2 (Isaac Gym–compatible).

Maps history of (obs, action) -> latent z_t. No Isaac Lab dependency.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Adaptation1DCNNCfg:
    """1D CNN adaptation module config for phase-2 online adaptation.
    
    Uses 1D convolutions over temporal history of observations and actions.
    Input: flattened history window (history_length * (obs_dim + action_dim))
    Output: latent extrinsics z_t (latent_dim)
    """
    
    in_channels: int  # obs_dim + action_dim per timestep
    history_length: int = 30  # Number of timesteps in history
    latent_dim: int = 8
    hidden_dims: tuple[int, ...] = (512, 256, 128)  # CNN channel dimensions


class Adaptation1DCNN(nn.Module):
    """1D CNN adaptation module that maps history window -> latent z_t."""

    def __init__(self, cfg: Adaptation1DCNNCfg):
        """Initialize 1D CNN adaptation module.
        
        Args:
            cfg: Adaptation1DCNNCfg configuration
        """
        super().__init__()
        self.cfg = cfg
        
        # Build 1D CNN: input (B, in_channels, history_length) -> output (B, latent_dim)
        layers = []
        prev_channels = cfg.in_channels
        
        for hidden_ch in cfg.hidden_dims:
            layers.append(nn.Conv1d(prev_channels, hidden_ch, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            prev_channels = hidden_ch
        
        self.conv_net = nn.Sequential(*layers)
        
        # Fully connected layer after convolution
        self.fc = nn.Linear(prev_channels * cfg.history_length, cfg.latent_dim)
    
    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            history: Shape (B, hist_dim) where hist_dim = history_length * in_channels
        
        Returns:
            Shape (B, latent_dim)
        """
        B = history.shape[0]
        
        # Reshape to (B, in_channels, history_length)
        x = history.view(B, self.cfg.history_length, self.cfg.in_channels)
        x = x.transpose(1, 2)  # (B, in_channels, history_length)
        
        # CNN forward
        x = self.conv_net(x)  # (B, hidden_channels[-1], history_length)
        
        # Flatten and FC
        x = x.reshape(B, -1)
        z = self.fc(x)
        
        return z
