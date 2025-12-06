"""
DTM Head Module Architecture: Base Class & MLP Implementation.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from abc import abstractmethod

from f5_tts.model.modules import TimestepEmbedding

class BaseDTMHead(nn.Module):
    """
    Base class for DTM Heads.
    Handles common logic: time embedding, input concatenation, and interface standardization.
    """
    def __init__(
        self,
        backbone_dim: int,
        mel_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.backbone_dim = backbone_dim
        self.mel_dim = mel_dim
        self.hidden_dim = hidden_dim
        
        # Common: Timestep embedding for microscopic time s
        self.time_embed = TimestepEmbedding(hidden_dim)
        
        # Common: Input projection (backbone + mel -> hidden)
        self.input_proj = nn.Linear(backbone_dim + mel_dim, hidden_dim)
        
        # Common: Output projection (hidden -> mel)
        self.output_proj = nn.Linear(hidden_dim, mel_dim)

    @abstractmethod
    def forward_net(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Abstract method for the core network computation.
        
        Args:
            x: Input features [N, hidden_dim] or [B, C, T] depending on subclass
            time_emb: Time embeddings [N, hidden_dim] or [B, hidden_dim]
            
        Returns:
            Processed features (same shape as x usually)
        """
        raise NotImplementedError

    def forward(
        self,
        h_t: torch.Tensor,  # [B, T, backbone_dim] or [N, backbone_dim] (N=batch*seq_len)
        y_s: torch.Tensor,  # [B, T, mel_dim] or [N, mel_dim]
        s: torch.Tensor,    # [B] or [N] or scalar
    ) -> torch.Tensor:
        """
        Standardized forward pass.
        Supports both 3D [B, T, D] and 2D [N, D] inputs.
        """
        # Detect input format
        is_3d = h_t.ndim == 3
        
        if is_3d:
            # 3D input: [B, T, D]
            b, t = h_t.shape[0], h_t.shape[1]
            
            # 1. Handle scalar or batch-level time s -> [B]
            if s.ndim == 0:
                s = s.repeat(b)
            elif s.shape[0] == b * t:
                # If s is already [B*T], take one per batch
                s = s.view(b, t)[:, 0]  # Use first timestep's s for each batch
            
            # 2. Time Embedding -> [B, hidden_dim]
            time_emb = self.time_embed(s)
            
            # 3. Input Concatenation & Projection
            x = torch.cat([h_t, y_s], dim=-1)  # [B, T, backbone + mel]
            x = self.input_proj(x)             # [B, T, hidden_dim]
            
            # 4. Core Network Forward (Subclass implementation)
            x = self.forward_net(x, time_emb)  # [B, T, hidden_dim]
            
            # 5. Output Projection
            v = self.output_proj(x)            # [B, T, mel_dim]
        else:
            # 2D input: [N, D] (original behavior for backward compatibility)
            N = h_t.shape[0]
            
            # 1. Handle scalar time s -> [N]
            if s.ndim == 0:
                s = s.repeat(N)
            
            # 2. Time Embedding -> [N, hidden_dim]
            time_emb = self.time_embed(s)
            
            # 3. Input Concatenation & Projection
            x = torch.cat([h_t, y_s], dim=-1)  # [N, backbone + mel]
            x = self.input_proj(x)             # [N, hidden_dim]
            
            # 4. Core Network Forward (Subclass implementation)
            x = self.forward_net(x, time_emb)
            
            # 5. Output Projection
            v = self.output_proj(x)            # [N, mel_dim]
        
        return v


