import torch
from transformers import ViTMAEConfig, ViTMAEForPreTraining

def create_mae_model(
    image_size=256,
    patch_size=32,
    embed_dim=256,
    encoder_depth=6,
    encoder_heads=8,
    decoder_embed_dim=128,
    decoder_depth=4,
    decoder_heads=4,
    mlp_ratio=4.0,
):
    """
    Creates a Masked Autoencoder (MAE) model with a Vision Transformer (ViT)
    backbone, initialized with random weights for training from scratch.
    
    Args:
        image_size (int): The size (height and width) of the input images.
        patch_size (int): The size of the patches to split the image into.
        embed_dim (int): The embedding dimension of the encoder.
        encoder_depth (int): The number of layers in the encoder.
        encoder_heads (int): The number of attention heads in the encoder.
        decoder_embed_dim (int): The embedding dimension of the decoder.
        decoder_depth (int): The number of layers in the decoder.
        decoder_heads (int): The number of attention heads in the decoder.
        mlp_ratio (float): The ratio for the MLP (feed-forward) layers' hidden size.

    Returns:
        A PyTorch model instance (ViTMAEForPreTraining) ready for training.
    """
    print("Creating a randomly initialized MAE model...")
    
    # 1. Define the model's architecture 
    config = ViTMAEConfig(
        image_size=image_size,
        patch_size=patch_size,
        hidden_size=embed_dim,
        num_hidden_layers=encoder_depth,
        num_attention_heads=encoder_heads,
        intermediate_size=int(embed_dim * mlp_ratio),
        decoder_hidden_size=decoder_embed_dim,
        decoder_num_hidden_layers=decoder_depth,
        decoder_num_attention_heads=decoder_heads,
        decoder_intermediate_size=int(decoder_embed_dim * mlp_ratio),
        mask_ratio=0.75,
        norm_pix_loss=True
    )

    # 2. Create the model from the configuration.
    model = ViTMAEForPreTraining(config)
    
    return model