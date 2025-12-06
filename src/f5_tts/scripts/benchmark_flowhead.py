import torch
import torch.nn as nn
import time
import numpy as np
import math

from f5_tts.template.flow_head import Matcha

# ==========================================
# 1. Shared Components & Utilities
# ==========================================

class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class SnakeBeta(nn.Module):
    def __init__(self, channels, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        super().__init__()
        
        self.alpha_logscale = alpha_logscale
        # --- 核心修改开始 ---
        # 我们需要 (1, C, 1) 的形状来匹配 Conv1D 的输出 (B, C, T)
        shape = (1, channels, 1)
        
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
        if self.alpha_logscale:
            alpha = torch.exp(self.alpha)
            beta = torch.exp(self.beta)
        else:
            alpha = self.alpha
            beta = self.beta
        
        # 现在 alpha 的形状是 (1, C, 1)，可以正确广播到 x 的 (B, C, T)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==========================================
# 2. Architectures
# ==========================================

# --- A. Baseline: 6-layer MLP ---
class BaselineMLP(nn.Module):
    def __init__(self, in_dim=1024, out_dim=100, hidden_dim=512, layers=6):
        super().__init__()
        self.net = nn.ModuleList([
            nn.Linear(in_dim if i==0 else hidden_dim, hidden_dim) 
            for i in range(layers)
        ])
        self.final = nn.Linear(hidden_dim, out_dim)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, T, C)
        for layer in self.net:
            x = self.act(layer(x))
        return self.final(x)

# --- B. ConvNeXt V2 ---
class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim, intermediate_dim, dilation=1):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x):
        input = x
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return input + x

class ConvNeXtModel(nn.Module):
    def __init__(self, in_dim=1024, out_dim=100, dim=512, layers=12):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, dim)
        self.layers = nn.ModuleList([
            ConvNeXtV2Block(dim, dim*4) for _ in range(layers)
        ])
        self.proj_out = nn.Linear(dim, out_dim)

    def forward(self, x):
        x = self.proj_in(x)
        for layer in self.layers:
            x = layer(x)
        return self.proj_out(x)

# --- C. ReFlow-TTS Decoder (WaveNet-style) ---

class ReFlowBlock(nn.Module):
    # 修复 1: 修改 init 参数以匹配 Model 中的调用方式
    # 原代码 Model 中是 ReFlowBlock(hidden, dilation=...)
    def __init__(self, channels, dilation):
        super().__init__()
        # 修复 2: 补全 nn. 前缀
        self.dilated_conv = nn.Conv1d(channels, 2 * channels, 3, padding=dilation, dilation=dilation)
        self.output_projection = nn.Conv1d(channels, 2 * channels, 1)

    # 修复 3: 移除未使用的 conditioner 和 diffusion_step 参数
    # (如果是为了测速 Benchmark，通常不需要这些；如果是真实模型，需要 Model 传入)
    def forward(self, x):
        y = self.dilated_conv(x)
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        
        # 返回残差流（用于下一层）和跳跃流（用于最终输出）
        return (x + residual) / math.sqrt(2.0), skip

class ReFlowModel(nn.Module):
    def __init__(self, in_dim=1024, out_dim=100, hidden=512, layers=6):
        super().__init__()
        self.proj_in = nn.Conv1d(in_dim, hidden, 1)
        self.layers = nn.ModuleList([
            # 这里的调用现在匹配了修改后的 Block 定义
            ReFlowBlock(hidden, dilation=2**(i%4)) for i in range(layers)
        ])
        self.proj_out = nn.Conv1d(hidden, out_dim, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.proj_in(x)
        
        # 修复 4: 正确处理 Skip Connection 聚合
        # WaveNet/DiffWave 结构的精髓是将所有层的 skip 加起来作为输出
        total_skip = 0
        for layer in self.layers:
            x, skip = layer(x) # 解包元组
            total_skip = total_skip + skip
            
        # 对 Skip Sum 进行缩放（保持方差稳定）并投影
        output = total_skip / math.sqrt(len(self.layers))
        
        return self.proj_out(output).transpose(1, 2)

# --- D. Matcha-TTS (ResNet1D + MLP) ---

class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (dilation * (kernel_size - 1)) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.act = SnakeBeta(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        
    def forward(self, x):
        res = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return res + x

class MatchaBlock(nn.Module):
    """
    组合块：ResBlock1D + MLP (Linear Implementation)
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.res_block = ResBlock1D(channels, kernel_size)
        
        # --- 坚持使用 Linear ---
        # 注意：不再使用 nn.Sequential，因为中间需要插播 transpose
        self.lin1 = nn.Linear(channels, channels * 2)
        
        # 你的 SnakeBeta 是针对 [B, C, T] 设计的 (参数形状 [1, C, 1])
        # 所以进入激活函数前必须把维度转回来
        self.act = SnakeBeta(channels * 2) 
        
        self.lin2 = nn.Linear(channels * 2, channels)

    def forward(self, x):
        # x shape: [B, C, T]
        
        # 1. ResBlock 分支 (保持 Channel First)
        x_res = self.res_block(x)
        
        # 2. MLP 分支 (Linear 需要 Channel Last)
        # [B, C, T] -> [B, T, C]
        y = x.transpose(1, 2) 
        
        # Linear 1: [B, T, C] -> [B, T, 2C]
        y = self.lin1(y) 
        
        # 转回 Channel First 给 SnakeBeta 吃: [B, T, 2C] -> [B, 2C, T]
        y = y.transpose(1, 2)
        y = self.act(y)
        
        # 再转回 Channel Last 给 Linear 2 吃: [B, 2C, T] -> [B, T, 2C]
        y = y.transpose(1, 2)
        
        # Linear 2: [B, T, 2C] -> [B, T, C]
        y = self.lin2(y)
        
        # 最后转回 Channel First 进行残差相加: [B, T, C] -> [B, C, T]
        y = y.transpose(1, 2)
        
        return x_res + y

class MatchaModel(nn.Module):
    def __init__(self, in_dim=1024, out_dim=100, base_dim=256):
        super().__init__()
        
        # 1. Input Projection
        self.proj_in = nn.Conv1d(in_dim, base_dim, 1)
        
        # 2. Encoder (Downsampling)
        # Level 1
        self.down1_block = MatchaBlock(base_dim)
        self.down1_sample = nn.Conv1d(base_dim, base_dim * 2, kernel_size=3, stride=2, padding=1)
        
        # Level 2
        self.down2_block = MatchaBlock(base_dim * 2)
        self.down2_sample = nn.Conv1d(base_dim * 2, base_dim * 4, kernel_size=3, stride=2, padding=1)
        
        # 3. Mid-Block (Bottleneck)
        self.mid_block1 = MatchaBlock(base_dim * 4)
        self.mid_block2 = MatchaBlock(base_dim * 4)
        
        # 4. Decoder (Upsampling)
        # Level 2 Up
        self.up2_sample = nn.ConvTranspose1d(base_dim * 4, base_dim * 2, kernel_size=4, stride=2, padding=1)
        self.up2_block = MatchaBlock(base_dim * 2) # Channel reduced after sum with skip
        
        # Level 1 Up
        self.up1_sample = nn.ConvTranspose1d(base_dim * 2, base_dim, kernel_size=4, stride=2, padding=1)
        self.up1_block = MatchaBlock(base_dim)
        
        # 5. Output Projection
        self.proj_out = nn.Conv1d(base_dim, out_dim, 1)
        
        # Initialize weights (Optional but recommended)
        self.apply(self._init_weights)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [B, Seq, Dim] -> [B, Dim, Seq]
        x = x.transpose(1, 2)
        
        # Input
        x = self.proj_in(x)
        
        # --- Encoder ---
        # Level 1
        x1 = self.down1_block(x)  # Skip connection 1
        x = self.down1_sample(x1) # Downsample
        
        # Level 2
        x2 = self.down2_block(x)  # Skip connection 2
        x = self.down2_sample(x2) # Downsample
        
        # --- Mid Block ---
        x = self.mid_block1(x)
        x = self.mid_block2(x)
        
        # --- Decoder ---
        # Level 2 Up
        x = self.up2_sample(x)
        # Handle shape mismatch due to odd sequence length (common U-Net issue)
        if x.shape[-1] != x2.shape[-1]:
            x = F.interpolate(x, size=x2.shape[-1], mode='linear', align_corners=False)
        x = x + x2 # Skip Connection Add
        x = self.up2_block(x)
        
        # Level 1 Up
        x = self.up1_sample(x)
        if x.shape[-1] != x1.shape[-1]:
            x = F.interpolate(x, size=x1.shape[-1], mode='linear', align_corners=False)
        x = x + x1 # Skip Connection Add
        x = self.up1_block(x)
        
        # Output
        x = self.proj_out(x)
        return x.transpose(1, 2)

# --- F. BigCodec (LSTM + ResCNN) ---
class ResCNNBlock(nn.Module):
    def __init__(self, channels=512, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(channels, channels, 1)
        
    def forward(self, x):
        res = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return res + x

class BigCodecModel(nn.Module):
    def __init__(self, in_dim=1024, out_dim=100, hidden=512, layers=6):
        super().__init__()
        self.proj_in = nn.Conv1d(in_dim, hidden, 7, padding=3)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True)
        self.blocks = nn.ModuleList([
            ResCNNBlock(hidden, kernel_size=7) for _ in range(layers)
        ])
        self.proj_out = nn.Conv1d(hidden, out_dim, 7, padding=3)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.proj_in(x)
        
        # LSTM part
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)
        
        for block in self.blocks:
            x = block(x)
        return self.proj_out(x).transpose(1, 2)

# --- G. DAC (Snake Conv) ---
class SnakeBlock(nn.Module):
    def __init__(self, channels=512, kernel_size=7, dilation=1):
        super().__init__()
        padding = (dilation * (kernel_size - 1)) // 2
        self.act1 = SnakeBeta(channels, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.act2 = SnakeBeta(channels, channels)
        self.conv2 = nn.Conv1d(channels, channels, 1)
        
    def forward(self, x):
        res = x
        x = self.act1(x)
        x = self.conv1(x)
        x = self.act2(x)
        x = self.conv2(x)
        return res + x

class DACModel(nn.Module):
    def __init__(self, in_dim=1024, out_dim=100, hidden=512, layers=12):
        super().__init__()
        self.proj_in = nn.Conv1d(in_dim, hidden, 7, padding=3)
        self.blocks = nn.ModuleList([
            SnakeBlock(hidden, dilation=3**((i%3))) for i in range(layers)
        ])
        self.proj_out = nn.Conv1d(hidden, out_dim, 7, padding=3)
        self.act_out = SnakeBeta(out_dim, out_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x)
        x = self.proj_out(x)
        x = self.act_out(x)
        return x.transpose(1, 2)

# ==========================================
# 3. Benchmarking
# ==========================================

def benchmark(model, input_tensor, name="Model", iterations=100):
    model.eval()
    model.cuda()
    input_tensor = input_tensor.cuda()

    # Assuming your batch size (B) is 100 based on the benchmark script
    B = input_tensor.shape[0]

    # Correction 1: Create a 1D tensor [B] on the correct device
    t = torch.full((B,), 0.5, device=input_tensor.device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_tensor, t=t)
    torch.cuda.synchronize()
    
    # Timing
    start = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(input_tensor, t = t)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / iterations * 100 # ms
    params = count_parameters(model) / 1e6 # Million
    
    print(f"{name:<20} | Params: {params:.2f}M | Time: {avg_time:.2f} s")
    return avg_time, params

# ==========================================
# 4. Run Comparisons
# ==========================================

if __name__ == "__main__":
    B, N, D = 100, 100, 1024
    dummy_input = torch.randn(B, N, D)
    
    print(f"Input Shape: {dummy_input.shape}")
    print("-" * 60)
    print(f"{'Model':<20} | {'Params (M)':<12} | {'Latency (ms)':<12}")
    print("-" * 60)

    # 4. Matcha (10 blocks, 384 hidden) -> ~21M
    benchmark(Matcha(), dummy_input, "Matcha-TTS")

    # 2. ConvNeXt V2 (12 layers, 512 hidden) -> ~25M
    # benchmark(ConvNeXtModel(dim=512, layers=15), dummy_input, "ConvNeXt V2")
    
    # 6. BigCodec (1 LSTM + 6 CNN, 512 hidden) -> ~24M
    # benchmark(BigCodecModel(hidden=512, layers=10), dummy_input, "BigCodec")
    
    # 7. DAC (12 blocks, 512 hidden, Snake) -> ~25M
    # benchmark(DACModel(hidden=512, layers=12), dummy_input, "DAC (Snake)")
    
    # 1. Baseline MLP (6 layers, 2048 hidden) -> ~25M
    # benchmark(BaselineMLP(hidden_dim=2048, layers=8), dummy_input, "MLP (Baseline)")
    

    
    # 3. ReFlow (20 layers, 256 fixed) -> ~2.7M (Under-parameterized per spec)
    # Note: To reach 25M with 256 dim, we'd need ~180 layers! 
    # I kept 20 layers as requested, so it's very fast/small.
    # benchmark(ReFlowModel(hidden=512, layers=15), dummy_input, "ReFlow-TTS")