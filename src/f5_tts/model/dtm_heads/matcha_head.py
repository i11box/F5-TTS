import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from f5_tts.model.modules import SinusPositionEmbedding, TimestepEmbedding, AdaLayerNorm
from f5_tts.model.dtm_heads.base_head import BaseDTMHead

# 1. 补全 SnakeBeta，适配 Linear 输入 (B, T, C)
class SnakeBeta(nn.Module):
    def __init__(self, channels, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        super().__init__()
        
        self.alpha_logscale = alpha_logscale
        # --- 核心修改开始 ---
        # 我们需要 (1, C, 1) 的形状来匹配 Conv1D 的输出 (B, C, T)
        shape = (1, 1, channels)
        
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(shape) * alpha)
            self.beta = nn.Parameter(torch.zeros(shape) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(shape) * alpha)
            self.beta = nn.Parameter(torch.ones(shape) * alpha)
        # --- 核心修改结束 ---

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-9

    def forward(self, x):
        alpha = self.alpha.exp()
        beta = self.beta.exp()
        return x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)

class Block1D(torch.nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        # print(dim)
        # print(dim_out)
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(dim, dim_out, 3, padding=1),
            torch.nn.GroupNorm(groups, dim_out),
            nn.Mish(),
        )
    def forward(self, x):
        # print(x.shape)
        return self.block(x)

class ResnetBlock1D(torch.nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        # 修正：Time MLP 接收独立的 time_emb_dim，投影到 dim_out
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(time_emb_dim, time_emb_dim), 
            nn.Mish(), 
            torch.nn.Linear(time_emb_dim, dim_out)
        )

        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)
        self.res_conv = torch.nn.Conv1d(dim, dim_out, 1) 

    def forward(self, x, time_emb):
        # x: [B, C, T]
        # time_emb: [B, time_emb_dim]
        h = self.block1(x)
        
        # Time Injection: [B, time_emb_dim] -> [B, dim_out] -> [B, dim_out, 1]
        time_emb_proj = self.time_mlp(time_emb).unsqueeze(-1)
        h = h + time_emb_proj
        
        h = self.block2(h)
        return h + self.res_conv(x)

class Downsample1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = torch.nn.Conv1d(dim, dim, 3, 2, 1)
    def forward(self, x):
        return self.conv(x)

class Upsample1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, 4, 2, 1)
    def forward(self, x):
        return self.conv(x)

class MLPBlock(nn.Module):
    """MLP block with AdaLayerNorm before MLP"""
    def __init__(self, dim, time_emb_dim=None):
        super().__init__()
        # If time_emb_dim is different from dim, add a projection layer
        self.time_emb_dim = time_emb_dim if time_emb_dim is not None else dim
        if self.time_emb_dim != dim:
            self.time_proj = nn.Linear(self.time_emb_dim, dim)
        else:
            self.time_proj = nn.Identity()
        
        self.norm = AdaLayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            SnakeBeta(dim * 2),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, dim]
            time_emb: Time embedding [batch, time_emb_dim]
        
        Returns:
            Output tensor [batch, seq_len, dim]
        """
        # Project time_emb to match dim
        time_emb_proj = self.time_proj(time_emb)  # [batch, dim]
        
        # AdaLN modulation
        x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm(x, time_emb_proj)
        
        # Apply MLP to normalized input
        ff_out = self.ff(x_norm)
        
        # Apply MLN modulation to FFN output
        ff_out = ff_out * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_out = ff_out * gate_mlp[:, None]
        
        # Residual connection
        return x + ff_out

# 4. 主模型
class Matcha(nn.Module):
    def __init__(
        self,
        in_channels=1024,
        out_channels=100,
        channels=(256, 256),
        dropout=0.05,
        n_blocks=1,
        num_mid_blocks=2,
        time_emb_dim=None,  # 新增：接收外部指定的 time embedding 维度
    ):
        super().__init__()
        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Time Embedding 维度：优先使用传入的，否则使用默认值
        if time_emb_dim is None:
            time_emb_dim = channels[0] * 4
        self.time_emb_dim = time_emb_dim

        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        # --- Down ---
        output_channel = in_channels
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            
            resnet = ResnetBlock1D(input_channel, output_channel, time_emb_dim=self.time_emb_dim)
            
            # MLP Blocks
            mlp_blocks = nn.ModuleList([self.get_block(output_channel) for _ in range(n_blocks)])
            
            downsample = Downsample1D(output_channel) if not is_last else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            self.down_blocks.append(nn.ModuleList([resnet, mlp_blocks, downsample]))

        # --- Mid ---
        mid_channel = channels[-1]
        for i in range(num_mid_blocks):
            resnet = ResnetBlock1D(mid_channel, mid_channel, time_emb_dim=self.time_emb_dim)
            mlp_blocks = nn.ModuleList([self.get_block(mid_channel) for _ in range(n_blocks)])
            self.mid_blocks.append(nn.ModuleList([resnet, mlp_blocks]))

        # --- Up ---
        # Reverse channels for upsampling
        up_channels = channels[::-1] # (256, 256) -> (256, 256)
        # We need to prepend the last channel because we just came from mid_blocks (which output channels[-1])
        # Input to first up block is: mid_output (channels[-1]) + skip (channels[-1])
        
        current_channel = channels[-1]
        
        for i in range(len(up_channels)):
            input_channel = current_channel
            output_channel = up_channels[i] # Target output
            
            # Skip connection comes from corresponding down block (in reverse order)
            # down_blocks are in order: channels[0], channels[1], ..., channels[-1]
            # up_blocks pop from hiddens in reverse: channels[-1], channels[-2], ..., channels[0]
            # So up_blocks[i] gets skip from down_blocks[-(i+1)]
            skip_channel = channels[-(i+1)]
            
            resnet = ResnetBlock1D(input_channel + skip_channel, output_channel, time_emb_dim=self.time_emb_dim)
            
            mlp_blocks = nn.ModuleList([self.get_block(output_channel) for _ in range(n_blocks)])
            
            is_last = i == len(up_channels) - 1
            upsample = Upsample1D(output_channel) if not is_last else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            
            self.up_blocks.append(nn.ModuleList([resnet, mlp_blocks, upsample]))
            current_channel = output_channel

        # Final block: project from channels[0] to out_channels
        self.final_block = Block1D(channels[0], out_channels)
        self.initialize_weights()

    # @staticmethod
    def get_block(self, dim):
        """Returns a simple MLP block (Linear -> SnakeBeta -> Linear)"""
        return MLPBlock(dim, time_emb_dim=self.time_emb_dim)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, t):
        """
        Args:
            x: Input tensor [B, C, T] where C=in_channels
            t: Time embedding [B, time_emb_dim]
        
        Returns:
            Output tensor [B, C_out, T] where C_out=out_channels (after final_block)
        """
        # Ensure x is in [B, C, T] format
        # Check if x is in [B, T, C] format by comparing channel dimension
        if x.dim() == 3 and x.shape[1] != self.in_channels and x.shape[-1] == self.in_channels:
            x = x.transpose(1, 2)  # [B, T, C] -> [B, C, T]
        
        # Validate input shape
        assert x.shape[1] == self.in_channels, \
            f"Expected input channels {self.in_channels}, got {x.shape[1]}"
        
        hiddens = []
        
        # --- Down ---
        for resnet, mlp_blocks, downsample in self.down_blocks:
            x = resnet(x, t)
            
            # MLP Part: (B, C, T) -> (B, T, C)
            x = rearrange(x, "b c t -> b t c")
            for mlp in mlp_blocks:
                x = mlp(x, t) # Residual
            x = rearrange(x, "b t c -> b c t")
            
            hiddens.append(x) # Store for skip
            x = downsample(x)

        # --- Mid ---
        for resnet, mlp_blocks in self.mid_blocks:
            x = resnet(x, t)
            x = rearrange(x, "b c t -> b t c")
            for mlp in mlp_blocks:
                x = mlp(x, t)
            x = rearrange(x, "b t c -> b c t")

        # --- Up ---
        for resnet, mlp_blocks, upsample in self.up_blocks:
            skip = hiddens.pop()
            
            # Handle shape mismatch (cropping/padding if needed)
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode='nearest')
            
            # Concat Skip Connection
            x = torch.cat([x, skip], dim=1)
            
            x = resnet(x, t)
            
            x = rearrange(x, "b c t -> b t c")
            for mlp in mlp_blocks:
                x = mlp(x, t)
            x = rearrange(x, "b t c -> b c t")
            
            x = upsample(x)

        # Final
        output = self.final_block(x)
            
        return output
    
class MatchaDTMHead(BaseDTMHead):
    """
    Matcha-Style U-Net Head wrapper to match DTMHead interface exactly.
    """
    def __init__(
        self,
        backbone_dim: int = 1024,  # DTMHead 标准参数
        mel_dim: int = 100,        # DTMHead 标准参数
        hidden_dim: int = 256,     # 内部 U-Net 通道数
        num_layers: int = 6,       # 对应原参数，用于控制U-Net深度
        dropout: float = 0.1,      # 对应原参数
        ff_mult: int = 4           # 对应原参数 (虽然这里不用)
    ):
        super().__init__(backbone_dim=backbone_dim, mel_dim=mel_dim, hidden_dim=hidden_dim)
        
        self.backbone_dim = backbone_dim
        self.mel_dim = mel_dim
        self.hidden_dim = hidden_dim
        
        # 2. 实例化你原本的 Matcha U-Net 结构
        # 注意：这里复用了你写的 Matcha 逻辑，但参数名做了适配
        self.unet = Matcha(
            in_channels=hidden_dim,   # 这里的 in_channels 对应 U-Net 内部流转的维度
            out_channels=hidden_dim,     # 最终输出 Mel 维度
            channels=(hidden_dim, hidden_dim * 2), # 简单配置两层，可根据 num_layers 调整
            dropout=dropout,
            n_blocks=1,
            num_mid_blocks=2,
            time_emb_dim=hidden_dim  # 传入 time embedding 维度，与 BaseDTMHead.time_embed 输出一致
        )

    def forward_net(
        self,
        x: torch.Tensor,  # [B, T, hidden_dim]
        time_emb: torch.Tensor,  # [B, hidden_dim]
    ) -> torch.Tensor:
        """
        Drop-in replacement forward pass.
        
        Args:
            x: Input tensor [B, T, hidden_dim]
            time_emb: Time embedding [B, hidden_dim]
        
        Returns:
            Output tensor [B, T, hidden_dim]
        """
        # 1. 维度检查
        if x.dim() == 2: 
            # 如果输入是 2D [N, D]，需要转换为 3D
            # 但我们无法知道 B 和 T，所以报错
            raise ValueError("MatchaDTMHead requires 3D input [B, T, D], but got 2D. "
                           "This should be handled by BaseDTMHead.")
        
        # 2. Transpose to [B, hidden_dim, T] for Conv1D
        x = x.transpose(1, 2)  # [B, T, hidden_dim] -> [B, hidden_dim, T]
        
        # 3. U-Net Forward
        # x: [B, hidden_dim, T]
        # time_emb: [B, hidden_dim]
        out = self.unet(x, time_emb)  # -> [B, hidden_dim, T]
        
        # 4. Transpose back to [B, T, hidden_dim]
        out = out.transpose(1, 2)  # [B, hidden_dim, T] -> [B, T, hidden_dim]
        
        return out