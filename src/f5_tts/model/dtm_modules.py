import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from f5_tts.model.modules import SinusPositionEmbedding, TimestepEmbedding

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
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(dim, dim_out, 3, padding=1),
            torch.nn.GroupNorm(groups, dim_out),
            nn.Mish(),
        )
    def forward(self, x):
        return self.block(x)

class ResnetBlock1D(torch.nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        # 修正：Time MLP 必须投影到 dim_out
        self.time_mlp = torch.nn.Sequential(nn.Mish(), torch.nn.Linear(time_emb_dim, dim_out))

        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)
        self.res_conv = torch.nn.Conv1d(dim, dim_out, 1) 

    def forward(self, x, time_emb):
        # x: [B, C, T]
        h = self.block1(x)
        
        # Time Injection: [B, C_out] -> [B, C_out, 1]
        time_emb = self.time_mlp(time_emb).unsqueeze(-1)
        h = h + time_emb
        
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
    ):
        super().__init__()
        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Time Embedding
        self.time_embeddings = SinusPositionEmbedding(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(in_channels, time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        # --- Down ---
        output_channel = in_channels
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            
            resnet = ResnetBlock1D(input_channel, output_channel, time_emb_dim=time_embed_dim)
            
            # MLP Blocks
            mlp_blocks = nn.ModuleList([self.get_block(output_channel) for _ in range(n_blocks)])
            
            downsample = Downsample1D(output_channel) if not is_last else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            self.down_blocks.append(nn.ModuleList([resnet, mlp_blocks, downsample]))

        # --- Mid ---
        mid_channel = channels[-1]
        for i in range(num_mid_blocks):
            resnet = ResnetBlock1D(mid_channel, mid_channel, time_emb_dim=time_embed_dim)
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
            
            # Skip connection adds channels
            # We assume skip connection has same channels as output_channel of corresponding down block
            # Actually, U-Net symmetric: skip comes from down block i.
            # Down path: in -> 256 (skip) -> 256 (skip)
            # Up path:   256 + 256(skip) -> 256 ...
            
            # Simplified logic: The skip connection channel count is usually equal to input_channel
            # if symmetric.
            skip_channel = input_channel # Assuming symmetric
            
            resnet = ResnetBlock1D(input_channel + skip_channel, output_channel, time_emb_dim=time_embed_dim)
            
            mlp_blocks = nn.ModuleList([self.get_block(output_channel) for _ in range(n_blocks)])
            
            is_last = i == len(up_channels) - 1
            upsample = Upsample1D(output_channel) if not is_last else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            
            self.up_blocks.append(nn.ModuleList([resnet, mlp_blocks, upsample]))
            current_channel = output_channel

        self.final_block = Block1D(channels[0], channels[0])
        self.final_proj = nn.Conv1d(channels[0], out_channels, 1)
        self.initialize_weights()

    @staticmethod
    def get_block(dim):
        """Returns a simple MLP block (Linear -> SnakeBeta -> Linear)"""
        return nn.Sequential(
            nn.Linear(dim, dim * 2),
            SnakeBeta(dim * 2), # Corrected SnakeBeta
            nn.Linear(dim * 2, dim)
        )

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, t):
        # 1. Embed Time
        t = self.time_mlp(t)

        # 2. Input Prep (Concat x and mu if needed, here assuming mu is part of input or ignored)
        # x: [B, C, T]
        if x.dim() == 3 and x.shape[-1] == self.in_channels:
            x = x.transpose(1, 2) # [B, T, C] -> [B, C, T]
        hiddens = []
        
        # --- Down ---
        for resnet, mlp_blocks, downsample in self.down_blocks:
            x = resnet(x, t)
            
            # MLP Part: (B, C, T) -> (B, T, C)
            x = rearrange(x, "b c t -> b t c")
            for mlp in mlp_blocks:
                x = x + mlp(x) # Residual
            x = rearrange(x, "b t c -> b c t")
            
            hiddens.append(x) # Store for skip
            x = downsample(x)

        # --- Mid ---
        for resnet, mlp_blocks in self.mid_blocks:
            x = resnet(x, t)
            x = rearrange(x, "b c t -> b t c")
            for mlp in mlp_blocks:
                x = x + mlp(x)
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
                x = x + mlp(x)
            x = rearrange(x, "b t c -> b c t")
            
            x = upsample(x)

        # Final
        x = self.final_block(x)
        output = self.final_proj(x)
            
        return output