"""
DTM (Distillation Transition Matching) Model.

This module implements the DTM framework for accelerating TTS inference.
It wraps a frozen DiT backbone with a trainable MLP head.

Training follows Algorithm 3 and inference follows Algorithm 4 from the guideline.
"""

from __future__ import annotations

from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from f5_tts.model.dtm_head import DTMHead
from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default,
    exists,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)


class DTM(nn.Module):
    """
    DTM (Distillation Transition Matching) model for accelerated TTS inference.
    
    Architecture:
    - Frozen DiT backbone (pretrained)
    - Trainable MLP head
    
    Training (Algorithm 3):
    - Sample discrete timestep t and compute X_t from X_0 and X_T
    - Extract frozen backbone features h_t
    - Sample microscopic time s and compute Y_s
    - Train head to predict velocity field
    
    Inference (Algorithm 4):
    - Loop over T global timesteps
    - For each timestep, solve ODE in microscopic space using head
    - Update global state
    
    Args:
        backbone: Frozen DiT backbone model
        head: Trainable DTM head
        global_timesteps: Number of global timesteps T (for inference)
        ode_solver_steps: Number of substeps for ODE solver
        ode_solver_method: ODE solver method ("euler" or "midpoint")
        mel_spec_module: Mel spectrogram module
        mel_spec_kwargs: Mel spectrogram configuration
        vocab_char_map: Vocabulary character map for tokenization
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        head: DTMHead | None = None,
        global_timesteps: int = 8,
        ode_solver_steps: int = 1,
        ode_solver_method: str = "euler",
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        vocab_char_map: dict[str:int] | None = None,
        audio_drop_prob: float = 0.3,
        cond_drop_prob: float = 0.2,
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
    ):
        super().__init__()
        
        # Frozen backbone
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        
        # Trainable head
        if head is None:
            head = DTMHead(
                backbone_dim=backbone.dim,
                mel_dim=mel_spec_kwargs.get('n_mel_channels', 100),
            )
        self.head = head
        
        # DTM parameters
        self.global_timesteps = global_timesteps
        self.T = global_timesteps
        self.ode_solver_steps = ode_solver_steps
        self.ode_solver_method = ode_solver_method
        
        # Classifier-free guidance parameters
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob
        
        # Conditional training parameters
        self.frac_lengths_mask = frac_lengths_mask
        
        # Mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        self.num_channels = self.mel_spec.n_mel_channels
        
        # Vocab map for tokenization
        self.vocab_char_map = vocab_char_map
        print(f'my ode {ode_solver_steps}')
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def extract_backbone_features(
        self,
        x: torch.Tensor,  # [batch, seq_len, mel_dim]
        cond: torch.Tensor,  # [batch, seq_len, mel_dim]
        text: torch.Tensor,  # [batch, text_len]
        time: torch.Tensor,  # [batch] or scalar
        mask: torch.Tensor | None = None,  # [batch, seq_len]
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cfg_infer: bool = False,  # CFG inference mode (only for inference)
    ) -> torch.Tensor:
        """
        Extract features from frozen backbone (before final projection).
        
        Args:
            drop_audio_cond: Whether to drop audio conditioning (for CFG)
            drop_text: Whether to drop text conditioning (for CFG)
            cfg_infer: If True, pack cond & uncond in batch dim (DiT-style CFG)
        
        Returns:
            h_t: Backbone features [batch, seq_len, backbone_dim] or [2*batch, seq_len, backbone_dim] if cfg_infer
        """
        # Get backbone hidden dimension
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Ensure time is batched
        if time.ndim == 0:
            time = time.repeat(batch_size)
        
        # CFG inference mode: pack cond & uncond forward (like DiT)
        if cfg_infer:
            # Conditional forward
            x_cond = self.backbone.get_input_embed(
                x, cond, text,
                drop_audio_cond=False,
                drop_text=False,
                cache=False,
                audio_mask=mask,
            )
            # Unconditional forward
            x_uncond = self.backbone.get_input_embed(
                x, cond, text,
                drop_audio_cond=True,
                drop_text=True,
                cache=False,
                audio_mask=mask,
            )
            # Concatenate in batch dimension
            x_embedded = torch.cat((x_cond, x_uncond), dim=0)
            t_emb = torch.cat((self.backbone.time_embed(time), self.backbone.time_embed(time)), dim=0)
            mask_packed = torch.cat((mask, mask), dim=0) if mask is not None else None
        else:
            # Normal forward
            x_embedded = self.backbone.get_input_embed(
                x, cond, text,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
                cache=False,
                audio_mask=mask,
            )
            t_emb = self.backbone.time_embed(time)
            mask_packed = mask
        
        # Forward through transformer blocks
        rope = self.backbone.rotary_embed.forward_from_seq_len(seq_len)
        
        h = x_embedded
        for block in self.backbone.transformer_blocks:
            h = block(h, t_emb, mask=mask_packed, rope=rope)
        
        # Apply final norm (but not projection)
        h_t = self.backbone.norm_out(h, t_emb)
        
        return h_t  # [batch, seq_len, backbone_dim] or [2*batch, seq_len, backbone_dim] if cfg_infer
    
    def forward(
        self,
        inp: torch.Tensor,  # mel or raw wave [batch, seq_len, mel_dim] or [batch, wave_len]
        text: torch.Tensor | list[str],  # [batch, text_len]
        *,
        lens: torch.Tensor | None = None,  # [batch]
        noise_scheduler = None # unused
    ):
        """
        Training forward pass (Algorithm 3).
        
        Args:
            inp: Input mel spectrogram or raw waveform
            text: Text input (tokens or strings)
            lens: Sequence lengths
        
        Returns:
            loss: MSE loss
            cond: Conditioning (for logging)
            pred: Predicted velocity field (for logging)
        """
        # Handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels
        
        batch_size, seq_len, dtype, device = *inp.shape[:2], inp.dtype, self.device
        
        # Handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch_size
        
        # Lens and mask
        if not exists(lens):
            lens = torch.full((batch_size,), seq_len, device=device)
        
        mask = lens_to_mask(lens, length=seq_len)
        
        # Get a random span to mask out for training conditionally (like CFM)
        frac_lengths = torch.zeros((batch_size,), device=device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)
        
        if exists(mask):
            rand_span_mask &= mask
        
        # X_T is the real mel spectrogram
        X_T = inp
        
        # X_0 is Gaussian noise
        X_0 = torch.randn_like(X_T)
        
        # Sample discrete timestep t uniformly from {1, ..., T-1}
        # We use continuous time in [0, 1] for compatibility with backbone
        # t = torch.randint(0, self.T, (batch_size,), device=device, dtype=torch.long)
        t_continuous = torch.rand((batch_size,), device=device, dtype=X_T.dtype)  # Normalize to [0, 1]
        
        # Compute X_t = (1 - t/T) * X_0 + (t/T) * X_T
        t_ratio = t_continuous.unsqueeze(-1).unsqueeze(-1) # [batch, 1, 1]
        X_t = (1 - t_ratio) * X_0 + t_ratio * X_T
        
        # Only predict what is within the random mask span for infilling (like CFM)
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(X_T), X_T)
        
        # Classifier-free guidance training with drop rate (like CFM)
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False
        
        # Extract frozen backbone features h_t
        # Training: directly forward once with drop flags (like CFM)
        # No need to pack cond & uncond during training
        with torch.no_grad():
            h_t = self.extract_backbone_features(
                x=X_t,
                cond=cond,
                text=text,
                time=t_continuous,
                mask=mask,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
                cfg_infer=False,  # Training时不用cfg_infer
            )
        
        # Compute target displacement Y = X_T - X_0
        Y = X_T - X_0
        
        # Flatten for token-level processing (following paper's reference implementation)
        # h_t: [batch, seq_len, backbone_dim] -> [batch*seq_len, backbone_dim]
        # Y: [batch, seq_len, mel_dim] -> [batch*seq_len, mel_dim]
        # t_continuous: [batch] -> [batch*seq_len]
        h_t_flat = h_t.reshape(batch_size * seq_len, -1)  # [batch*seq_len, backbone_dim]
        Y_flat = Y.reshape(batch_size * seq_len, -1)  # [batch*seq_len, mel_dim]
        
        # Also flatten the mask for later use
        rand_span_mask_flat = rand_span_mask.reshape(-1)  # [batch*seq_len]
        
        # Sample microscopic time s uniformly from [0, 1] for each token
        s = torch.rand((batch_size * seq_len,), device=device, dtype=dtype)
        
        # Sample microscopic noise
        Y_noise = torch.randn_like(Y_flat)
        
        # Compute Y_s = (1 - s) * Y_noise + s * Y
        s_expand = s.unsqueeze(-1)  # [batch*seq_len, 1]
        Y_s = (1 - s_expand) * Y_noise + s_expand * Y_flat
        
        # Forward through trainable head (now accepts flattened input)
        v_pred = self.head(h_t_flat, Y_s, s)
        
        # Compute target velocity: Y - Y_noise
        v_target = Y_flat - Y_noise
        
        # Compute MSE loss with mask (like CFM, only on masked region)
        loss = F.mse_loss(v_pred, v_target, reduction='none')  # [batch*seq_len, mel_dim]
        
        # Apply rand_span_mask to only compute loss on the region to be predicted
        loss = loss[rand_span_mask_flat]
        loss = loss.mean()
        
        return loss, cond, v_pred
    
    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,  # [batch, cond_len, mel_dim] or [batch, wave_len]
        text: torch.Tensor | list[str],  # [batch, text_len]
        duration: int | torch.Tensor,  # target duration
        *,
        lens: torch.Tensor | None = None,  # [batch]
        steps: int | None = None,  # override global_timesteps
        cfg_strength: float = 1.0,  # classifier-free guidance strength
        seed: int | None = None,
        max_duration: int = 4096,
        sway_sampling_coef = None, # useless
        vocoder: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        """
        Inference sampling (Algorithm 4).
        
        Args:
            cond: Conditioning mel spectrogram or waveform
            text: Text input
            duration: Target duration
            lens: Conditioning lengths
            steps: Number of global timesteps (override self.T)
            seed: Random seed
            max_duration: Maximum duration
            vocoder: Vocoder for converting mel to waveform
        
        Returns:
            out: Generated mel spectrogram [batch, duration, mel_dim]
            trajectory: Trajectory of generation (for visualization)
        """
        self.eval()
        
        # Handle raw wave conditioning
        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels
        
        cond = cond.to(next(self.parameters()).dtype)
        
        batch_size, cond_seq_len, device = *cond.shape[:2], cond.device
        
        if not exists(lens):
            lens = torch.full((batch_size,), cond_seq_len, device=device, dtype=torch.long)
        
        # Handle text
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch_size
        
        # Handle duration
        cond_mask = lens_to_mask(lens)
        
        if isinstance(duration, int):
            duration = torch.full((batch_size,), duration, device=device, dtype=torch.long)
        
        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()
        
        # Pad conditioning
        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))
        
        if batch_size > 1:
            mask = lens_to_mask(duration)
        else:
            mask = None
        
        # Use specified steps or default global_timesteps
        T = steps if steps is not None else self.T
        
        # Initialize X_0 with Gaussian noise
        X = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            X.append(torch.randn(dur, self.num_channels, device=device, dtype=cond.dtype))
        X = pad_sequence(X, padding_value=0, batch_first=True)  # [batch, max_dur, mel_dim]
        
        # Algorithm 4: Loop over global timesteps
        trajectory = [X.clone()]
        # Generate time steps t_0, t_1, ..., t_N using Sway Sampling
        # This is where Sway Sampling belongs in DTM (scheduling the global trajectory)
        t_steps = torch.linspace(0, 1, T + 1, device=device, dtype=cond.dtype)
        
        # Apply Sway Sampling to the global time schedule
        if sway_sampling_coef is not None:
             t_steps = t_steps + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_steps) - 1 + t_steps)
        
        # DTM Loop
        for i in range(T):
            t_curr = t_steps[i]
            t_next = t_steps[i+1]
            dt = t_next - t_curr # Dynamic step size based on Sway Sampling

            # Current global time (continuous)
            t_continuous = torch.full((batch_size,), t_curr, device=device, dtype=cond.dtype)
            
            # Extract backbone features h_t with CFG (DiT-style batch packing)
            if cfg_strength < 1e-5:
                # No CFG, single forward pass
                h_t = self.extract_backbone_features(
                    x=X,
                    cond=step_cond,
                    text=text,
                    time=t_continuous,
                    mask=mask,
                    drop_audio_cond=False,
                    drop_text=False,
                    cfg_infer=False,
                )
                
                # Solve ODE for Y using the head (Inner loop micro-step)
                # Note: Inner loop usually uses standard uniform steps or simple midpoint
                # No Sway Sampling needed here as this is a local approximation
                Y_0 = torch.randn_like(X)
                
                def ode_fn(s, y):
                    s_batch = torch.full((batch_size,), s.item(), device=device, dtype=cond.dtype)
                    return self.head(h_t, y, s_batch)
                
                s_span = torch.linspace(0, 1, self.ode_solver_steps + 1, device=device, dtype=cond.dtype)
                Y_trajectory = odeint(
                    ode_fn,
                    Y_0,
                    s_span,
                    method=self.ode_solver_method,
                )
                Y_final = Y_trajectory[-1]
            else:
                # With CFG: pack cond & uncond in batch dimension (DiT-style)
                h_t_cfg = self.extract_backbone_features(
                    x=X,
                    cond=step_cond,
                    text=text,
                    time=t_continuous,
                    mask=mask,
                    cfg_infer=True,  # Pack cond & uncond forward
                )
                # Split conditional and unconditional features
                h_t_cond, h_t_uncond = torch.chunk(h_t_cfg, 2, dim=0)
                
                # Solve ODE with CFG
                Y_0 = torch.randn_like(X)
                
                def ode_fn_cfg(s, y):
                    s_batch = torch.full((batch_size,), s.item(), device=device, dtype=cond.dtype)
                    # Predict with both conditional and unconditional
                    v_cond = self.head(h_t_cond, y, s_batch)
                    v_uncond = self.head(h_t_uncond, y, s_batch)
                    # Apply CFG
                    return v_cond + (v_cond - v_uncond) * cfg_strength
                
                s_span = torch.linspace(0, 1, self.ode_solver_steps + 1, device=device, dtype=cond.dtype)
                # Inner loop usually doesn't need sway sampling, standard linear is fine
                Y_trajectory = odeint(
                    ode_fn_cfg,
                    Y_0,
                    s_span,
                    method=self.ode_solver_method,
                )
                Y_final = Y_trajectory[-1]
            
            # Update global state: X_{t+1} = X_t + dt * Y_final
            # --- MODIFIED: Use dynamic dt from Sway Sampling ---
            # Standard DTM uses 1/T, but with Sway Sampling we use (t_next - t_curr)
            X = X + dt * Y_final
            
            trajectory.append(X.clone())
        
        # Final output
        out = X
        out = torch.where(cond_mask, cond, out)  # Keep conditioning part
        
        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)
        
        trajectory = torch.stack(trajectory, dim=0)  # [T+1, batch, seq_len, mel_dim]
        
        return out, trajectory

    @torch.no_grad()
    def dumm_sample(
        self,
        cond: torch.Tensor,
        text: torch.Tensor | list[str],
        duration: int | torch.Tensor,
        *,
        lens: torch.Tensor | None = None,
        steps: int | None = None,
        cfg_strength: float = 1.0,
        seed: int | None = None,
        max_duration: int = 4096,
        vocoder: Callable[[torch.Tensor], torch.Tensor] | None = None,
        sway_sampling_coef = None
    ):
        """
        Debug function: Sample using DiT's original projection head instead of DTM Head.
        This validates if the backbone features and DTM sampling logic are correct.
        """
        print("\n[DEBUG] Running dummy_sample with frozen DiT projection...")
        self.eval()
        
        # Handle raw wave conditioning
        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels
        
        cond = cond.to(next(self.parameters()).dtype)
        
        batch_size, cond_seq_len, device = *cond.shape[:2], cond.device
        
        if not exists(lens):
            lens = torch.full((batch_size,), cond_seq_len, device=device, dtype=torch.long)
        
        # Handle text
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch_size
        
        # Handle duration
        cond_mask = lens_to_mask(lens)
        
        if isinstance(duration, int):
            duration = torch.full((batch_size,), duration, device=device, dtype=torch.long)
        
        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()
        
        # Pad conditioning
        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))
        
        if batch_size > 1:
            mask = lens_to_mask(duration)
        else:
            mask = None
        
        # Use specified steps or default global_timesteps
        T = steps if steps is not None else self.T
        
        # Initialize X_0 with Gaussian noise
        X = []
        for dur in duration:
            if exists(seed): torch.manual_seed(seed)
            X.append(torch.randn(dur, self.num_channels, device=device, dtype=cond.dtype))
        X_0 = pad_sequence(X, padding_value=0, batch_first=True)
        
        # 定义 ODE 函数 (Backbone Forward)
        def ode_fn(t, x):
            # t 是标量，广播到 batch
            t_batch = torch.full((batch_size,), t.item(), device=device, dtype=x.dtype)
            
            # 使用 cfg_infer=True 进行高效推理
            if cfg_strength < 1e-5:
                # Unconditional / Single forward
                v_pred = self.backbone(
                    x=x,
                    cond=step_cond,
                    text=text,
                    time=t_batch,
                    mask=mask,
                    drop_audio_cond=False,
                    drop_text=False
                )
                return v_pred
            else:
                # CFG Forward (DiT 内部同时跑 cond 和 uncond)
                # 注意：这要求 Backbone 支持 cfg_infer 参数，且内部逻辑是 cat([x, x])
                v_out = self.backbone(
                    x=x,
                    cond=step_cond,
                    text=text,
                    time=t_batch,
                    mask=mask,
                    cfg_infer=True # 这一步很关键，复用了原版的高效推理逻辑
                )
                v_cond, v_uncond = torch.chunk(v_out, 2, dim=0)
                
                # 使用你提供的公式 (F5-TTS 风格)
                # v_final = v_cond + (v_cond - v_uncond) * cfg
                return v_cond + (v_cond - v_uncond) * cfg_strength

        # 定义时间步
        t_span = torch.linspace(0, 1, T + 1, device=device, dtype=cond.dtype)
        
        # 如果你想加上 Sway Sampling (可选)
        sway_sampling_coef = -1.0
        t_span = t_span + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)

        # --- 3. 执行 ODE 求解 ---
        # method='midpoint' 或 'euler'，midpoint 精度更高推荐用于 Debug
        trajectory = odeint(ode_fn, X_0, t_span, method='euler')
        
        # 取最后一步作为结果
        out = trajectory[-1]
        
        # 强制回填 Ref Audio (In-filling 修正)
        out = torch.where(cond_mask, cond, out)
        
        # --- 4. Vocoder 解码 ---
        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)
            
        return out, trajectory
    
    def dum_forward(
        self,
        inp: torch.Tensor,  # mel or raw wave [batch, seq_len, mel_dim] or [batch, wave_len]
        text: torch.Tensor | list[str],  # [batch, text_len]
        *,
        lens: torch.Tensor | None = None,  # [batch]
        noise_scheduler = None # unused
    ):
        """
        Training forward pass (Algorithm 3).
        
        Args:
            inp: Input mel spectrogram or raw waveform
            text: Text input (tokens or strings)
            lens: Sequence lengths
        
        Returns:
            loss: MSE loss
            cond: Conditioning (for logging)
            pred: Predicted velocity field (for logging)
        """
        # Handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels
        
        batch_size, seq_len, dtype, device = *inp.shape[:2], inp.dtype, self.device
        
        # Handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch_size
        
        # Lens and mask
        if not exists(lens):
            lens = torch.full((batch_size,), seq_len, device=device)
        
        mask = lens_to_mask(lens, length=seq_len)
        
        # Get a random span to mask out for training conditionally (like CFM)
        frac_lengths = torch.zeros((batch_size,), device=device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)
        
        if exists(mask):
            rand_span_mask &= mask
        
        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((batch_size,), dtype=dtype, device=self.device)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # apply mask will use more memory; might adjust batchsize or batchsampler long sequence threshold
        pred = self.backbone(
            x=φ, cond=cond, text=text, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text, mask=mask
        )

    
        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]
        
        dummy_loss = 0.0 * sum(p.sum() for p in self.head.parameters())
        loss = loss + dummy_loss

        return loss.mean(), cond, pred