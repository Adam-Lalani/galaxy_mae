import torch
import wandb
import matplotlib.pyplot as plt
import numpy as np

def denormalize(image, mean, std):
    """Denormalizes a tensor image with mean and standard deviation."""
    image = image.clone()
    mean = torch.tensor(mean, device=image.device).view(3, 1, 1)
    std = torch.tensor(std, device=image.device).view(3, 1, 1)
    image.mul_(std).add_(mean)
    image = torch.clamp(image, 0, 1)
    return image

def log_reconstruction_image(model, image, train_mean, train_std, device, epoch):
    """
    Generates a reconstruction of a single image, plots it, and logs it to W&B.
    
    The HuggingFace MAE model reconstructs masked patches. We combine:
    - Visible patches: from the original image
    - Masked patches: from the model's reconstruction
    """
    model.eval()
    with torch.no_grad():
        image_batch = image.unsqueeze(0).to(device)
        
        model_to_visualize = model.module if hasattr(model, 'module') else model

        outputs = model_to_visualize(pixel_values=image_batch)
        mask = outputs.mask.detach()  # Shape: (batch, num_patches), 1 = masked, 0 = visible

        # --- Get Original Patches ---
        original_patches = model_to_visualize.patchify(image_batch)  # (batch, num_patches, patch_size^2 * 3)
        
        # --- Create Masked Image (zero out masked patches) ---
        mask_expanded = mask.unsqueeze(-1)  # (batch, num_patches, 1)
        masked_patches = original_patches * (1 - mask_expanded)  # Keep visible, zero masked
        masked_image_tensor = model_to_visualize.unpatchify(masked_patches).squeeze(0)
        
        # --- Create Full Reconstruction (model's prediction for ALL patches) ---
        # outputs.logits contains predictions for ALL patches
        pred_patches = outputs.logits.detach()  # (B, N, P)

        if getattr(model_to_visualize.config, "norm_pix_loss", True):
            # compute per-patch stats from targets (i.e., original_patches)
            patch_mean = original_patches.mean(dim=-1, keepdim=True)            # (B, N, 1)
            patch_var  = original_patches.var(dim=-1, keepdim=True, unbiased=False)
            patch_std  = (patch_var + 1e-6).sqrt()

            # undo the per-patch normalization
            pred_patches = pred_patches * patch_std + patch_mean  # back to dataset-normalized pixel space
        
        full_reconstruction_tensor = model_to_visualize.unpatchify(pred_patches).squeeze(0)

        # --- Hybrid: visible from original + masked from reconstruction ---
        hybrid_patches = original_patches * (1 - mask_expanded) + pred_patches * mask_expanded
        hybrid_reconstruction_tensor = model_to_visualize.unpatchify(hybrid_patches).squeeze(0)

        # --- Plotting ---
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f'MAE Reconstruction at Epoch {epoch}', fontsize=16)

        original_vis = denormalize(image.cpu(), train_mean, train_std).permute(1, 2, 0).numpy()
        masked_vis = denormalize(masked_image_tensor.cpu(), train_mean, train_std).permute(1, 2, 0).numpy()
        full_recon_vis = denormalize(full_reconstruction_tensor.cpu(), train_mean, train_std).permute(1, 2, 0).numpy()
        hybrid_vis = denormalize(hybrid_reconstruction_tensor.cpu(), train_mean, train_std).permute(1, 2, 0).numpy()

        axs[0].imshow(original_vis); axs[0].set_title('Original'); axs[0].axis('off')
        axs[1].imshow(masked_vis); axs[1].set_title('Masked (75%)'); axs[1].axis('off')
        axs[2].imshow(full_recon_vis); axs[2].set_title('Full Reconstruction'); axs[2].axis('off')
        axs[3].imshow(hybrid_vis); axs[3].set_title('Hybrid (Visible + Recon)'); axs[3].axis('off')
        
        wandb.log({"Reconstructions": wandb.Image(fig, caption=f"Epoch {epoch}")}, step=epoch)
        
        plt.close(fig)

    model.train()