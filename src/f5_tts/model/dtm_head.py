"""
DTM (Distillation Transition Matching) MLP Head module.

This module implements a lightweight MLP head that works with frozen DiT backbone
to accelerate TTS inference by predicting displacement vectors in flow space.

Based on the DTM framework described in the guideline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from f5_tts.model.modules import TimestepEmbedding, AdaLayerNorm, FeedForward


class DTMHeadBlock(nn.Module):
    """
    A single block in the DTM Head.
    
    Uses AdaLN to inject microscopic time conditioning and FFN for transformation.
    """
    
    def __init__(
        self,
        dim: int,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.norm = AdaLayerNorm(dim)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout)
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, dim]
            time_emb: Time embedding [batch, dim]
        
        Returns:
            Output tensor [batch, seq_len, dim]
        """
        # AdaLN modulation
        x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm(x, time_emb)
        
        # Since we're using MLP-only (no attention), we use x_norm directly
        # and apply gate to the FFN output
        ff_out = self.ff(x_norm)
        
        # Apply MLN modulation to FFN output
        ff_out = ff_out * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_out = ff_out * gate_mlp[:, None]
        
        # Residual connection
        return x + ff_out


class DTMHead(nn.Module):
    """
    DTM MLP Head for predicting velocity field (displacement vectors).
    
    Architecture:
    - Input: Concatenation of backbone features h_t (backbone_dim) and flow state y_s (mel_dim)
    - Input projection: (backbone_dim + mel_dim) -> hidden_dim
    - N layers of DTMHeadBlock with AdaLN (time conditioning) and FFN
    - Output projection: hidden_dim -> mel_dim
    
    Args:
        backbone_dim: Dimension of backbone features (e.g., 1024 for F5-TTS)
        mel_dim: Dimension of mel spectrogram (e.g., 100)
        hidden_dim: Hidden dimension for MLP layers (e.g., 512)
        num_layers: Number of MLP blocks (e.g., 6)
        ff_mult: FFN expansion multiplier (e.g., 4 for 512->2048->512)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        backbone_dim: int = 1024,
        mel_dim: int = 100,
        hidden_dim: int = 512,
        num_layers: int = 6,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.backbone_dim = backbone_dim
        self.mel_dim = mel_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Timestep embedding for microscopic time s
        # Output dimension should match hidden_dim for AdaLayerNorm
        self.time_embed = TimestepEmbedding(hidden_dim)
        
        # Input projection: [h_t, y_s] -> hidden_dim
        # h_t has shape [batch, seq_len, backbone_dim]
        # y_s has shape [batch, seq_len, mel_dim]
        # We need to project concatenated features to hidden_dim
        self.input_proj = nn.Linear(backbone_dim + mel_dim, hidden_dim)
        
        # Main body: N layers of MLP blocks
        self.blocks = nn.ModuleList([
            DTMHeadBlock(
                dim=hidden_dim,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Output projection: hidden_dim -> mel_dim
        self.output_proj = nn.Linear(hidden_dim, mel_dim)
        
        # Initialize output projection to zero (for stability)
        nn.init.constant_(self.output_proj.weight, 0)
        nn.init.constant_(self.output_proj.bias, 0)
    
    def forward(
        self,
        h_t: torch.Tensor,  # Backbone features [N, backbone_dim] where N=batch*seq_len
        y_s: torch.Tensor,  # Flow state [N, mel_dim] where N=batch*seq_len
        s: torch.Tensor,  # Microscopic time [N] where N=batch*seq_len, or scalar
    ) -> torch.Tensor:
        """
        Forward pass of DTM Head.
        
        This follows the paper's reference implementation where inputs are flattened
        to process each token independently with its own microscopic time.
        
        Args:
            h_t: Backbone features [N, backbone_dim] where N=batch*seq_len
            y_s: Current flow state [N, mel_dim] where N=batch*seq_len
            s: Microscopic time [N] where N=batch*seq_len, or scalar
        
        Returns:
            Predicted velocity field [N, mel_dim] where N=batch*seq_len
        """
        num_tokens = h_t.shape[0]
        
        # Handle scalar time
        if s.ndim == 0:
            s = s.repeat(num_tokens)
        
        # Time embedding [N, hidden_dim]
        time_emb = self.time_embed(s)
        
        # Concatenate backbone features and flow state
        # h_t: [N, backbone_dim]
        # y_s: [N, mel_dim]
        x = torch.cat([h_t, y_s], dim=-1)  # [N, backbone_dim + mel_dim]
        
        # Input projection
        x = self.input_proj(x)  # [N, hidden_dim]
        
        # Add sequence dimension for compatibility with AdaLayerNorm
        # AdaLayerNorm expects [batch, seq_len, dim], so we treat each token as a batch
        x = x.unsqueeze(1)  # [N, 1, hidden_dim]
        
        # Apply MLP blocks with time conditioning
        for block in self.blocks:
            x = block(x, time_emb)  # [N, 1, hidden_dim]
        
        # Remove sequence dimension
        x = x.squeeze(1)  # [N, hidden_dim]
        
        # Output projection to predict velocity field
        v = self.output_proj(x)  # [N, mel_dim]
        
        return v

