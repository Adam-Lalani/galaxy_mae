import torch
import torch.nn as nn # Import nn for DataParallel
import time
import wandb
from data import get_dataloaders
from model import create_mae_model
from training_utils import train_mae_one_epoch, evaluate_linear_probe


if __name__ == '__main__':
    # --- HYPERPARAMETERS ---
    config = {
        # General Training Settings
        "epochs": 150,
        "batch_size": 64,
        "num_workers": 4,
        "lr_mae": 1e-4,
        "probe_epochs": 10,
        
        # MAE Model Configuration - Slightly larger for a longer run
        "image_size": 256,
        "patch_size": 32,
        "embed_dim": 384,
        "encoder_depth": 8,
        "encoder_heads": 8,
        "decoder_embed_dim": 192,
        "decoder_depth": 6,
        "decoder_heads": 6
    }   

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # --- 1. SETUP ---
    wandb.init(project="galaxy-mae-pretraining", config=config)
    
    print(f"Using device: {DEVICE}")
    
    mae_loader, probe_train_loader, probe_test_loader = get_dataloaders(
        batch_size=wandb.config.batch_size, 
        image_size=wandb.config.image_size, 
        num_workers=wandb.config.num_workers
    )
    
    mae_model = create_mae_model(
        image_size=wandb.config.image_size, patch_size=wandb.config.patch_size, 
        embed_dim=wandb.config.embed_dim, encoder_depth=wandb.config.encoder_depth, 
        encoder_heads=wandb.config.encoder_heads, decoder_embed_dim=wandb.config.decoder_embed_dim, 
        decoder_depth=wandb.config.decoder_depth, decoder_heads=wandb.config.decoder_heads
    )

    if DEVICE == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        mae_model = nn.DataParallel(mae_model)

    mae_model.to(DEVICE)

    model_to_optimize = mae_model.module if isinstance(mae_model, nn.DataParallel) else mae_model
    mae_optimizer = torch.optim.AdamW(model_to_optimize.parameters(), lr=wandb.config.lr_mae)

    # --- 2. MAIN TRAINING LOOP ---
    print("\nStarting MAE Pre-training and Evaluation...")
    start_time = time.time()
    for epoch in range(1, wandb.config.epochs + 1):
        print(f"\n{'='*25} Epoch {epoch}/{wandb.config.epochs} {'='*25}")
        
        # a. Train MAE for one epoch
        avg_mae_loss = train_mae_one_epoch(mae_model, mae_loader, mae_optimizer, DEVICE)
        print(f"Epoch {epoch} | Average MAE Loss: {avg_mae_loss:.4f}")
        
        # b. Create a dictionary to hold all metrics for this epoch
        log_metrics = {
            "epoch": epoch,
            "mae_loss": avg_mae_loss,
        }
        
        # c. Evaluate with Linear Probe periodically (every 10 epochs)
        if epoch % 10 == 0:
            encoder_to_probe = mae_model.module.vit if isinstance(mae_model, nn.DataParallel) else mae_model.vit
            probe_accuracy = evaluate_linear_probe(
                encoder=encoder_to_probe, 
                train_loader=probe_train_loader, 
                test_loader=probe_test_loader, 
                device=DEVICE,
                probe_epochs=wandb.config.probe_epochs
            )
            # Add the accuracy to our log dictionary
            log_metrics["probe_accuracy"] = probe_accuracy
        
        # d. Log all collected metrics to W&B in a single call
        wandb.log(log_metrics)
        
        # e. Save a checkpoint periodically
        if epoch % 10 == 0:
            checkpoint_path = f"mae_galaxy_epoch_{epoch}.pth"
            # Save the state_dict of the underlying model, not the DataParallel wrapper
            model_to_save = mae_model.module if isinstance(mae_model, nn.DataParallel) else mae_model
            torch.save(model_to_save.state_dict(), checkpoint_path)
            # Log the checkpoint artifact to W&B
            wandb.save(checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    end_time = time.time()