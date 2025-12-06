from __future__ import annotations
import torch
import torch.nn as nn

from f5_tts.model.dtm_heads.base_head import BaseDTMHead
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

class MLPDTMHead(BaseDTMHead):
    """
    The original MLP-based DTM Head implementation.
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
        # Initialize base class
        super().__init__(backbone_dim, mel_dim, hidden_dim)
        
        self.num_layers = num_layers
        
        # MLP Specific: Stack of DTMHeadBlocks
        self.blocks = nn.ModuleList([
            DTMHeadBlock(
                dim=hidden_dim,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
    def forward_net(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        MLP specific forward pass.
        Input x: [N, hidden_dim] (Flattened tokens)
        """
        # Add dummy sequence dimension for AdaLayerNorm compatibility
        # AdaLayerNorm expects [Batch, Seq, Dim], here we treat N tokens as Batch=N, Seq=1
        x = x.unsqueeze(1) # [N, 1, hidden_dim]
        
        for block in self.blocks:
            x = block(x, time_emb)
            
        # Remove dummy dimension
        x = x.squeeze(1)   # [N, hidden_dim]
        
        return x