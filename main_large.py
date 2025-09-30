import torch
import torch.nn as nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import time
import wandb
import argparse
from data import get_dataloaders
from mae_model import create_mae_model
from train_and_eval import train_mae_one_epoch, evaluate_linear_probe, fine_tune_and_evaluate
from visualization import log_reconstruction_image 

if __name__ == '__main__':

    config = {
        
        # Pretraining
        "epochs": 1,           # 400 
        "batch_size": 64,        
        "num_workers": 4,
        "lr_mae": 1e-4,        
        "probe_epochs": 10,
        "warmup_epochs": 1,     # 40
        "min_lr": 1e-6,
        "weight_decay": 0.05,
        "adam_betas": (0.9, 0.95),
        "seed": 42,
        
        # Finetuning
        "finetune_epochs": 1,  # 50
        "finetune_lr": 1e-5,
        "finetune_warmup_epochs": 1,  # 5
        "finetune_min_lr": 1e-6,
        
        # MAE Model Configuration - ViT-Base
        "image_size": 256,
        "patch_size": 16,        
        "embed_dim": 768,         
        "encoder_depth": 12,     
        "encoder_heads": 12,     
        "decoder_embed_dim": 512, 
        "decoder_depth": 8,      
        "decoder_heads": 16      
    }   

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    torch.manual_seed(config["seed"])

    # --- SETUP ---
    # Start a new W&B run with a descriptive name
    wandb.init(project="galaxy-mae-pretraining", config=config, name="vit-base-400-epochs", entity="adam_lalani-brown-university")
    
    print(f"Using device: {DEVICE}")
    
    mae_loader, probe_train_loader, probe_test_loader, train_mean, train_std = get_dataloaders(
        batch_size=wandb.config.batch_size, 
        image_size=wandb.config.image_size, 
        num_workers=wandb.config.num_workers
    )
    
    vis_image = next(iter(probe_test_loader))['pixel_values'][0]
    
    mae_model = create_mae_model(**{k: v for k, v in config.items() if k in create_mae_model.__code__.co_varnames})

    if DEVICE == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        mae_model = nn.DataParallel(mae_model)
    mae_model.to(DEVICE)

    model_to_optimize = mae_model.module if isinstance(mae_model, nn.DataParallel) else mae_model
    mae_optimizer = torch.optim.AdamW(model_to_optimize.parameters(), lr=wandb.config.lr_mae, weight_decay=wandb.config.weight_decay, betas=wandb.config.adam_betas)
    
    scheduler = SequentialLR(
        mae_optimizer, 
        schedulers=[
            LinearLR(mae_optimizer, start_factor=0.01, total_iters=wandb.config.warmup_epochs),
            CosineAnnealingLR(mae_optimizer, T_max=wandb.config.epochs - wandb.config.warmup_epochs, eta_min=wandb.config.min_lr)
        ], 
        milestones=[wandb.config.warmup_epochs]
    )


    # --- MAIN TRAINING LOOP ---
    print("\nStarting MAE Pre-training and Evaluation...")
    start_time = time.time()
    for epoch in range(1, wandb.config.epochs + 1):
        print(f"\n{'='*25} Epoch {epoch}/{wandb.config.epochs} {'='*25}")
        
        avg_mae_loss = train_mae_one_epoch(mae_model, mae_loader, mae_optimizer, DEVICE)
        
        log_metrics = { "mae_loss": avg_mae_loss }
        
        scheduler.step()
        log_metrics["learning_rate"] = scheduler.get_last_lr()[0]
        wandb.log(log_metrics, step=epoch)
        
        # Save every 25 epochs
        if epoch % 1 == 0:  # FIX: 25
            print(f"--- Epoch {epoch}: Logging reconstruction image and saving checkpoint ---")
            
            # Log reconstruction image
            log_reconstruction_image(mae_model, vis_image, train_mean, train_std, DEVICE, epoch)
            
            # Save checkpoint
            checkpoint_path = f"mae_galaxy_vit_base_epoch_{epoch}.pth"
            model_to_save = mae_model.module if isinstance(mae_model, nn.DataParallel) else mae_model
            torch.save(model_to_save.state_dict(), checkpoint_path)
            wandb.save(checkpoint_path)

    # --- FINAL EVALUATION ---
    print(f"\n{'='*50}")
    print("Training completed! Running final linear probe evaluation...")
    print(f"{'='*50}")
    
    # Extract the encoder for linear probing
    encoder_to_probe = mae_model.module.vit if isinstance(mae_model, nn.DataParallel) else mae_model.vit
    
    # Run linear probe evaluation
    final_probe_accuracy = evaluate_linear_probe(
        encoder=encoder_to_probe, 
        train_loader=probe_train_loader, 
        test_loader=probe_test_loader, 
        device=DEVICE,
        probe_epochs=wandb.config.probe_epochs
    )
    
    # Run Full Fine-tuning Evaluation
    final_finetune_accuracy = fine_tune_and_evaluate(
        encoder=encoder_to_probe, train_loader=probe_train_loader,
        test_loader=probe_test_loader, device=DEVICE,
        finetune_epochs=wandb.config.finetune_epochs,
        lr=wandb.config.finetune_lr,
        warmup_epochs=wandb.config.finetune_warmup_epochs,
        min_lr=wandb.config.finetune_min_lr
    )
    
    # Log final results
    wandb.log({
        "final_probe_accuracy": final_probe_accuracy,
        "final_finetune_accuracy": final_finetune_accuracy,
    })
    
    print(f"\n--- Final Results ---")
    print(f"Linear Probe Accuracy: {final_probe_accuracy:.4f}%")
    print(f"Fine-tuning Accuracy: {final_finetune_accuracy:.4f}%")
    print(f"Total Experiment Time: {(time.time() - start_time) / 3600:.2f} hours")

    wandb.finish()
