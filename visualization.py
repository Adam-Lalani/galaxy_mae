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
    """
    model.eval()
    with torch.no_grad():
        image_batch = image.unsqueeze(0).to(device)
        
        model_to_visualize = model.module if hasattr(model, 'module') else model

        outputs = model_to_visualize(pixel_values=image_batch)
        mask = outputs.mask.detach() # mask has 1s for masked patches, 0s for visible

        # --- Create the Masked Image ---
        patches = model_to_visualize.patchify(image_batch)
        
        # --- FIX: Use robust element-wise multiplication to apply the mask ---
        masked_patches = patches * (1 - mask.unsqueeze(-1))
        
        masked_image_tensor = model_to_visualize.unpatchify(masked_patches).squeeze(0)
        
        # --- Get the Reconstructed Image ---
        reconstructed_image_tensor = model_to_visualize.unpatchify(outputs.logits).squeeze(0)

        # --- Plotting ---
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'MAE Reconstruction at Epoch {epoch}', fontsize=16)

        original_vis = denormalize(image.cpu(), train_mean, train_std).permute(1, 2, 0).numpy()
        masked_vis = denormalize(masked_image_tensor.cpu(), train_mean, train_std).permute(1, 2, 0).numpy()
        recon_vis = denormalize(reconstructed_image_tensor.cpu(), train_mean, train_std).permute(1, 2, 0).numpy()

        axs[0].imshow(original_vis); axs[0].set_title('Original'); axs[0].axis('off')
        axs[1].imshow(masked_vis); axs[1].set_title('Masked (75%)'); axs[1].axis('off')
        axs[2].imshow(recon_vis); axs[2].set_title('Reconstructed'); axs[2].axis('off')
        
        wandb.log({"Reconstructions": wandb.Image(fig, caption=f"Epoch {epoch}")})
        
        plt.close(fig)

    model.train()