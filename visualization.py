import torch
import wandb
import matplotlib.pyplot as plt
import numpy as np

def denormalize(image, mean, std):
    """Denormalizes a tensor image with mean and standard deviation."""
    # Clone to avoid modifying the original tensor
    image = image.clone()
    
    # Reshape mean and std to be broadcastable
    mean = torch.tensor(mean, device=image.device).view(3, 1, 1)
    std = torch.tensor(std, device=image.device).view(3, 1, 1)
    
    image.mul_(std).add_(mean)
    image = torch.clamp(image, 0, 1)
    return image

def log_reconstruction_image(model, image, train_mean, train_std, device, epoch):
    """
    Generates a reconstruction of a single image, plots it, and logs it to W&B.
    
    Args:
        model (nn.Module): The MAE model.
        image (torch.Tensor): The single image tensor to reconstruct (must be on CPU).
        train_mean (list): The mean of the training dataset for denormalization.
        train_std (list): The std of the training dataset for denormalization.
        device (str): The device to run the model on.
        epoch (int): The current epoch, for titling the plot.
    """
    model.eval()
    with torch.no_grad():
        image_batch = image.unsqueeze(0).to(device)
        
        # The model might be wrapped in DataParallel, so access the module if needed
        model_to_visualize = model.module if hasattr(model, 'module') else model

        # Perform a forward pass to get the outputs
        outputs = model_to_visualize(pixel_values=image_batch)
        mask = outputs.mask.detach()

        # --- Create the Masked Image ---
        patches = model_to_visualize.patchify(image_batch)
        patches[mask] = 0 # Set masked patches to black
        masked_image_tensor = model_to_visualize.unpatchify(patches).squeeze(0)
        
        # --- Get the Reconstructed Image ---
        reconstructed_image_tensor = model_to_visualize.unpatchify(outputs.logits).squeeze(0)

        # --- Plotting ---
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'MAE Reconstruction at Epoch {epoch}', fontsize=16)

        # Denormalize for visualization (move tensors to CPU for numpy conversion)
        original_vis = denormalize(image.cpu(), train_mean, train_std).permute(1, 2, 0).numpy()
        masked_vis = denormalize(masked_image_tensor.cpu(), train_mean, train_std).permute(1, 2, 0).numpy()
        recon_vis = denormalize(reconstructed_image_tensor.cpu(), train_mean, train_std).permute(1, 2, 0).numpy()

        # Plot Original, Masked, and Reconstructed
        axs[0].imshow(original_vis); axs[0].set_title('Original'); axs[0].axis('off')
        axs[1].imshow(masked_vis); axs[1].set_title('Masked (75%)'); axs[1].axis('off')
        axs[2].imshow(recon_vis); axs[2].set_title('Reconstructed'); axs[2].axis('off')
        
        # Log the figure to Weights & Biases under a single key for easy comparison
        wandb.log({"Reconstructions": wandb.Image(fig, caption=f"Epoch {epoch}")})
        
        plt.close(fig)

    model.train()

