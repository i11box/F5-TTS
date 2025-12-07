"""
Training script for DTM (Distillation Transition Matching) model.

This script loads a pretrained frozen DiT backbone and trains a lightweight
MLP head to accelerate inference from 32 steps to 4-8 steps.
"""

import os
from importlib.resources import files

import hydra
import torch
from omegaconf import OmegaConf

from f5_tts.model import DiT, DTM, DTMHead, Trainer
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer


os.chdir(str(files("f5_tts").joinpath("../..")))  # change working directory to root of project

# 筛掉无法分词的数据
def is_text_valid(text, vocab_set):
    if vocab_set is None: 
        return True
    return set(text).issubset(vocab_set)

def load_backbone_from_checkpoint(
    checkpoint_path: str,
    model_cls,
    model_arch: dict,
    text_num_embeds: int,
    mel_dim: int,
    device: str = "cpu",
):
    """
    Load pretrained DiT backbone from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_cls: Model class (e.g., DiT)
        model_arch: Model architecture config
        text_num_embeds: Number of text embeddings (will be inferred from checkpoint if available)
        mel_dim: Mel dimension
        device: Device to load to
    
    Returns:
        tuple: (backbone model, actual vocab size used)
    """
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Backbone checkpoint not found at {checkpoint_path}. "
            f"Please provide a valid pretrained F5-TTS checkpoint path in the config."
        )
    
    print(f"Loading pretrained backbone from {checkpoint_path}")
    
    # First, load checkpoint to infer vocab size
    # Check checkpoint format and load accordingly
    ckpt_type = checkpoint_path.split(".")[-1].lower()
    if ckpt_type == "safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError(
                "safetensors library is required to load .safetensors checkpoints. "
                "Install it with: pip install safetensors"
            )
        # safetensors.load_file returns a dict directly (state dict)
        checkpoint = load_file(checkpoint_path, device=device)
        # safetensors files typically contain the state dict directly
        # But handle wrapped formats if present
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "ema_model_state_dict" in checkpoint:
            state_dict = checkpoint["ema_model_state_dict"]
        else:
            # Direct state dict (most common for safetensors)
            # Check if keys have ema_model prefix
            state_dict = checkpoint
    elif ckpt_type == "pt":
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        
        # Extract model state dict (handle different checkpoint formats)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "ema_model_state_dict" in checkpoint:
            # Use EMA model if available
            state_dict = checkpoint["ema_model_state_dict"]
        else:
            # Assume checkpoint is the state dict itself
            state_dict = checkpoint
    else:
        raise ValueError(
            f"Unsupported checkpoint format: {ckpt_type}. "
            f"Supported formats: .pt, .safetensors"
        )
    
    # Remove ema_model prefix if present (e.g., "ema_model.transformer.xxx" -> "transformer.xxx")
    if any(k.startswith("ema_model.") for k in state_dict.keys()):
        state_dict = {k.replace("ema_model.", ""): v for k, v in state_dict.items()}
    
    # Remove wrapped model prefix if present (e.g., "transformer.xxx" -> "xxx")
    # DTM expects the backbone without CFM wrapper
    if any(k.startswith("transformer.") for k in state_dict.keys()):
        state_dict = {k.replace("transformer.", ""): v for k, v in state_dict.items()}
    
    # Filter out non-model keys (mel_spec, EMA metadata, etc.)
    state_dict = {
        k: v for k, v in state_dict.items() 
        if not k.startswith("mel_spec.") and k not in ["initted", "step"]
    }

    backbone = model_cls(**model_arch, text_num_embeds=text_num_embeds, mel_dim=mel_dim)
    # Load state dict
    missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys in backbone: {missing_keys[:5]}...")  # show first 5
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")
    
    print("Backbone loaded successfully!")
    
    return backbone


@hydra.main(version_base="1.3", config_path=str(files("f5_tts").joinpath("configs")), config_name=None)
def main(model_cfg):
    """Main training function for DTM."""
    
    # Get model configuration
    model_cls = hydra.utils.get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    backbone_arch = model_cfg.model.backbone_arch
    head_arch = model_cfg.model.head_arch
    dtm_config = model_cfg.model.dtm
    tokenizer = model_cfg.model.tokenizer
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
    train_together = model_cfg.model.train_together
    
    exp_name = f"{model_cfg.model.name}_{mel_spec_type}_{tokenizer}_{model_cfg.datasets.name}"
    wandb_resume_id = None
    
    # Set text tokenizer
    if tokenizer != "custom":
        tokenizer_path = model_cfg.datasets.name
    else:
        tokenizer_path = model_cfg.model.tokenizer_path
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)
    
    # Load pretrained frozen backbone
    backbone_checkpoint_path = model_cfg.ckpts.backbone_checkpoint_path
    backbone = load_backbone_from_checkpoint(
        checkpoint_path=backbone_checkpoint_path,
        model_cls=model_cls,
        model_arch=backbone_arch,
        text_num_embeds=vocab_size,
        mel_dim=model_cfg.model.mel_spec.n_mel_channels,
        device="cpu",
    )
    
    # Create DTM head
    head = DTMHead(**head_arch)
    
    # Create DTM model
    model = DTM(
        backbone=backbone,
        head=head,
        global_timesteps=dtm_config.global_timesteps,
        ode_solver_steps=dtm_config.ode_solver_steps,
        ode_solver_method=dtm_config.ode_solver_method,
        mel_spec_kwargs=model_cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
        audio_drop_prob=dtm_config.get('audio_drop_prob', 0.3),
        cond_drop_prob=dtm_config.get('cond_drop_prob', 0.2),
        frac_lengths_mask=tuple(dtm_config.get('frac_lengths_mask', [0.7, 1.0])),
    )
    
    # Verify backbone is frozen
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = total_params - trainable_params
    
    print(f"\nModel Parameter Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Frozen parameters (backbone): {frozen_params:,}")
    print(f"  Trainable parameters (head): {trainable_params:,}")
    print(f"  Trainable ratio: {trainable_params/total_params*100:.2f}%\n")
    
    # Verify only head parameters are trainable
    backbone_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    head_trainable = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    
    assert backbone_trainable == 0, "ERROR: Backbone parameters should be frozen!"
    assert head_trainable == trainable_params, "ERROR: Only head parameters should be trainable!"
    print("✓ Verification passed: Backbone is frozen, only head is trainable.\n")
    
    # Prepare model config dict for logging
    model_cfg_dict = {
        "model_name": model_cfg.model.name,
        "backbone": model_cfg.model.backbone,
        "global_timesteps": dtm_config.global_timesteps,
        "ode_solver_steps": dtm_config.ode_solver_steps,
        "ode_solver_method": dtm_config.ode_solver_method,
        "head_hidden_dim": head_arch.hidden_dim,
        "head_name": head_arch.name,
        "epochs": model_cfg.optim.epochs,
        "learning_rate": model_cfg.optim.learning_rate,
        "num_warmup_updates": model_cfg.optim.num_warmup_updates,
        "batch_size_per_gpu": model_cfg.datasets.batch_size_per_gpu,
        "batch_size_type": model_cfg.datasets.batch_size_type,
        "max_samples": model_cfg.datasets.max_samples,
        "grad_accumulation_steps": model_cfg.optim.grad_accumulation_steps,
        "max_grad_norm": model_cfg.optim.max_grad_norm,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
    }
    
    # Initialize trainer
    trainer = Trainer(
        model,
        epochs=model_cfg.optim.epochs,
        learning_rate=model_cfg.optim.learning_rate,
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=str(files("f5_tts").joinpath(f"../../{model_cfg.ckpts.save_dir}")),
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu,
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=model_cfg.datasets.max_samples,
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps,
        max_grad_norm=model_cfg.optim.max_grad_norm,
        logger=model_cfg.ckpts.logger,
        wandb_project="DTM-TTS",
        wandb_run_name=exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_updates=model_cfg.ckpts.last_per_updates,
        log_samples=model_cfg.ckpts.log_samples,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=model_cfg.model.vocoder.is_local,
        local_vocoder_path=model_cfg.model.vocoder.local_path,
        model_cfg_dict=model_cfg_dict,
    )
    
    # 在 Trainer 初始化后重建优化器（支持分组学习率）
    if train_together:
        trainer.optimizer = trainer.accelerator.prepare(
            torch.optim.AdamW([
                {'params': model.head.parameters(), 'lr': model_cfg.optim.learning_rate},  # head 用配置的 lr
                {'params': model.backbone.parameters(), 'lr': 5e-6}  # backbone 极低 lr（解冻后生效）
            ], weight_decay=0.01)
        )
    
    # Load dataset
    train_dataset = load_dataset(
        model_cfg.datasets.name,
        tokenizer,
        mel_spec_kwargs=model_cfg.model.mel_spec
    )
    
    # Train
    print(f"Starting DTM training with {model_cfg.datasets.name} dataset...")
    print(f"Global timesteps T: {dtm_config.global_timesteps}")
    print(f"ODE solver: {dtm_config.ode_solver_method} with {dtm_config.ode_solver_steps} steps\n")
    
    trainer.train(
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()

