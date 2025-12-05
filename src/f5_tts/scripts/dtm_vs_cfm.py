
import torch
import torch.nn.functional as F
from f5_tts.model.backbones.dit import DiT
from f5_tts.model.cfm import CFM
from f5_tts.model.dtm import DTM
from torchdiffeq import odeint

def test_dtm_vs_cfm():
    print("Initializing Models...")
    # Setup small models for testing
    dim = 64
    mel_dim = 32
    # IMPORTANT: Enable long_skip_connection to test the potential bug
    dit = DiT(
        dim=dim, 
        depth=2, 
        heads=4, 
        mel_dim=mel_dim, 
        text_num_embeds=20, 
        long_skip_connection=True
    )
    
    # CFM wraps DiT
    cfm = CFM(
        transformer=dit,
        mel_spec_kwargs={"n_mel_channels": mel_dim}
    )
    
    # DTM wraps the SAME DiT instance (frozen in DTM usually, but here we share it)
    dtm = DTM(
        backbone=dit,
        mel_spec_kwargs={"n_mel_channels": mel_dim}
    )
    
    # Set to eval to disable dropout for deterministic comparison
    dit.eval()
    cfm.eval()
    dtm.eval()
    
    print("\n--- 1. Comparing Backbone Feature Extraction ---")
    # Create dummy inputs
    B, N = 2, 50
    x = torch.randn(B, N, mel_dim)
    cond = torch.randn(B, N, mel_dim)
    text = torch.randint(0, 20, (B, N))
    time = torch.rand(B)
    # Create a mask (some True, some False to test masking logic)
    mask = torch.ones(B, N, dtype=torch.bool)
    # mask[:, -10:] = False # Mask last 10 frames
    
    with torch.no_grad():
        # 1. Run DiT (Standard CFM path)
        # DiT.forward returns the final projected output (mel_dim)
        out_dit = dit(
            x=x, 
            cond=cond, 
            text=text, 
            time=time, 
            mask=mask,
            drop_audio_cond=False,
            drop_text=False
        )
        
        # 2. Run DTM extract_backbone_features
        # This returns the hidden state BEFORE projection (dim)
        h_dtm = dtm.extract_backbone_features(
            x=x, 
            cond=cond, 
            text=text, 
            time=time, 
            mask=mask,
            drop_audio_cond=False,
            drop_text=False
        )
        
        # 3. Manually project DTM features using DiT's output projection
        # If extract_backbone_features is correct, this should match out_dit
        out_dtm_projected = dtm.head(h_dtm)
        
        # Compare
        diff = (out_dit - out_dtm_projected).abs().max()
        print(f"DiT Output Shape: {out_dit.shape}")
        print(f"DTM Features Shape: {h_dtm.shape}")
        print(f"DTM Projected Shape: {out_dtm_projected.shape}")
        print(f"Max Difference: {diff.item()}")
        
        if diff > 1e-5:
            print("\n[!] CRITICAL DIFFERENCE FOUND in Backbone Features!")
            print("Explanation: DTM.extract_backbone_features is NOT producing the same features as DiT.forward.")
            print("This confirms that the input processing or transformer block loop in DTM is missing something present in DiT.")
            if hasattr(dit, 'long_skip_connection') and dit.long_skip_connection is not None:
                print("Note: DiT has long_skip_connection enabled. Check if DTM handles this.")
        else:
            print("\n[OK] Backbone features match.")

    print("\n--- 2. Comparing Training Logic (Loss Calculation) ---")
    # We will simulate one training step with IDENTICAL random values
    
    # Fix random inputs for "sampling" t and x0
    # In CFM/DTM forward, these are generated internally. 
    # To compare, we must manually compute the loss using the same formula outside the classes.
    
    # Common inputs
    x1 = torch.randn(B, N, mel_dim) # Target (Clean)
    x0 = torch.randn(B, N, mel_dim) # Noise
    t = torch.rand(B) # Time
    
    # CFM Logic Simulation
    # phi = (1 - t) * x0 + t * x1
    # flow = x1 - x0
    t_expanded = t.view(B, 1, 1)
    phi = (1 - t_expanded) * x0 + t_expanded * x1
    target_flow = x1 - x0
    
    # Run DiT (CFM)
    pred_cfm = dit(x=phi, cond=cond, text=text, time=t, mask=mask)
    loss_cfm = F.mse_loss(pred_cfm, target_flow, reduction='none')
    loss_cfm = loss_cfm[mask].mean()
    
    print(f"CFM Manual Loss: {loss_cfm.item()}")
    
    # DTM Logic Simulation
    # DTM uses extract_backbone_features -> Head -> Loss
    # If we assume DTM Head is just a linear projection (as user said they replaced it),
    # let's use DiT's proj_out as the "Head" for this test.
    
    h_dtm_train = dtm.extract_backbone_features(x=phi, cond=cond, text=text, time=t, mask=mask)
    pred_dtm = dtm.head(h_dtm_train) # Using DiT proj as surrogate for DTM head
    
    loss_dtm = F.mse_loss(pred_dtm, target_flow, reduction='none')
    loss_dtm = loss_dtm[mask].mean()
    
    print(f"DTM Manual Loss: {loss_dtm.item()}")
    
    diff_loss = abs(loss_cfm.item() - loss_dtm.item())
    print(f"Loss Difference: {diff_loss}")
    
    print("\n--- 3. Comparing Sampling/Inference Logic ---")
    # Setup for sampling
    n_steps = 10  # Use a small number of steps for testing
    # Generate identical initial noise
    x0 = torch.randn(B, 50, mel_dim)
    
    # Define time steps (Linear 0 to 1)
    # Note: If you use Sway Sampling, ensure both models use the exact same t_span
    t_span = torch.linspace(0, 1, n_steps + 1)
    
    print(f"Testing with {n_steps} Euler steps...")

    # --- A. CFM Sampling Simulation (Standard Euler) ---
    # x_{t+1} = x_t + v_pred * dt
    def fn(t, x):
        # at each step, conditioning is fixed
        # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

        # predict flow (cond)
        pred = cfm.transformer(
            x=x, 
            cond=cond, 
            text=text, 
            time=t, 
            mask=mask,
            drop_audio_cond=False,
            drop_text=False
        )
        return pred

    # noise input
    # to make sure batch inference result is same with different batch size, and for sure single inference
    # still some difference maybe due to convolutional layers

    trajectory = odeint(fn, x0, t_span, method='euler')

    out_1 = trajectory[-1]

    # --- B. DTM Sampling Simulation (Deterministic Mode) ---
    # x_{t+1} = x_t + Y_pred * dt
    # Key: For this test, we MUST use 'dit' as the surrogate head for DTM 
    # to prove the LOOP logic is identical.
    trajectory = [x0.clone()]
    # Generate time steps t_0, t_1, ..., t_N using Sway Sampling
    # This is where Sway Sampling belongs in DTM (scheduling the global trajectory)
    t_steps = torch.linspace(0, 1, n_steps + 1)
    
    # Apply Sway Sampling to the global time schedule
    
    def vector_field(t, x):
        t_batch = t.expand(B)
        h_t_cfg = dtm.extract_backbone_features(
            x=x, 
            cond=cond, 
            text=text, 
            time=t_batch, 
            mask=mask,
            drop_audio_cond=False,
            drop_text=False
        )
        v_final = dtm.head(h_t_cfg)
        
        return v_final
            
    # 3. 执行积分
    # odeint 会自动计算每一步的 dt = t_steps[i+1] - t_steps[i]
    # 并调用 vector_field(t_steps[i], x_i)
    trajectory = odeint(
        vector_field,
        x0,              # 初始状态 X0 (高斯噪声)
        t_steps,        # 积分的时间点
        method='euler'  # 使用 Euler 方法 (同你手写的逻辑)
        # method='midpoint' # 如果想更高精度，可以换成这个
    )
    
    # trajectory shape: [T+1, batch, seq_len, mel_dim]
    # 取最后一个时间点作为结果
    out_2 = trajectory[-1]

    # --- Compare Trajectories ---
    final_diff = (out_1 - out_2).abs().max()
    print(f"Final Sampling Difference: {final_diff.item()}")
    
    if final_diff > 1e-5:
        print("\n[!] Sampling Logic Mismatch!")
        print("Possible causes:")
        print("1. 'dt' calculation is different (e.g. DTM using fixed 1/T vs dynamic dt)")
        print("2. Time input 't' passed to model is different")
    else:
        print("\n[OK] Sampling logic is mathematically equivalent (Euler).")
    
    if diff_loss > 1e-5:
        print("\n[!] Loss calculation differs (likely due to feature extraction difference above).")
    else:
        print("\n[OK] Loss calculation logic is consistent.")

if __name__ == "__main__":
    test_dtm_vs_cfm()
