# Galaxy MAE: Masked Autoencoder for Galaxy Classification

This repository implements a Masked Autoencoder (MAE) for self-supervised learning on galaxy images from the Galaxy10-DECALS dataset. The model learns rich visual representations by reconstructing masked patches of galaxy images, which can then be used for downstream classification tasks.

## Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd MAE_galaxy
pip install -r requirements.txt
```

### 2. Run Training

```bash
python main.py
```

That's it! The script will automatically:
- Download the Galaxy10-DECALS dataset from Hugging Face
- Calculate dataset statistics (mean/std) for proper normalization
- Train a MAE model with the default configuration
- Evaluate the learned representations using linear probing
- Log everything to Weights & Biases

## Project Structure

```
MAE_galaxy/
├── main.py                 # Main training script
├── mae_model.py           # MAE model definition using Hugging Face transformers
├── data.py                # Dataset loading and preprocessing
├── train_and_eval.py      # Training and evaluation functions
├── visualization.py       # Reconstruction visualization utilities
├── visualize_checkpoint.ipynb  # Jupyter notebook for checkpoint analysis
├── run_oscar.slurm        # SLURM script for cluster training
├── models/                # Directory for saved model checkpoints
│   └── Mae_Galaxy_Vit_Base_Epoch_400.pth
└── requirements.txt       # Python dependencies
```

## Configuration

The model uses the following default configuration (modifiable in `main.py`):

```python
config = {
    # Training Settings
    "epochs": 150,
    "batch_size": 64,
    "lr_mae": 1e-4,
    "probe_epochs": 10,
    "warmup_epochs": 10,
    
    # Model Architecture
    "image_size": 256,
    "patch_size": 32,
    "embed_dim": 384,
    "encoder_depth": 8,
    "encoder_heads": 8,
    "decoder_embed_dim": 192,
    "decoder_depth": 6,
    "decoder_heads": 6,
    "mlp_ratio": 4.0
}
```

## Features

### Self-Supervised Pre-training
- **Masked Autoencoder**: Learns representations by reconstructing 75% masked patches
- **Vision Transformer Backbone**: Uses ViT architecture for robust feature learning
- **No Data Augmentation**: Clean reconstruction without augmentation artifacts

### Evaluation Methods
- **Linear Probing**: Freezes encoder, trains only a linear classifier
- **Fine-tuning**: Jointly trains encoder and classifier with data augmentation
- **Reconstruction Visualization**: Visual comparison of original, masked, and reconstructed images

### Training Features
- **Multi-GPU Support**: Automatic DataParallel for multiple GPUs
- **Mixed Precision**: Optional mixed precision training for efficiency
- **Learning Rate Scheduling**: Warmup + cosine annealing
- **Checkpointing**: Saves model every 25 epochs
- **Experiment Tracking**: Full Weights & Biases integration

## Usage Examples

### Basic Training
```bash
python main.py
```

### Resume from Checkpoint
```bash
python main.py --resume_checkpoint models/Mae_Galaxy_Vit_Base_Epoch_400.pth --start_epoch 401
```

### Resume W&B Run
```bash
python main.py --resume_wandb_id your-run-id
```

### Cluster Training (SLURM)
```bash
sbatch run_oscar.slurm
```

## Visualization

### Jupyter Notebook
Use `visualize_checkpoint.ipynb` to:
- Load trained checkpoints
- Visualize reconstruction quality
- Compare different model configurations
- Analyze learned representations

### Weights & Biases
The training automatically logs:
- Loss curves and learning rates
- Reconstruction visualizations
- Linear probe accuracy
- Model checkpoints as artifacts

## Dataset

**Galaxy10-DECALS**: 17,736 galaxy images from the Dark Energy Camera Legacy Survey
- **Classes**: 10 different galaxy morphologies
- **Resolution**: Variable, resized to 256×256
- **Preprocessing**: Center crop (80×80) → resize → normalize
- **Split**: Train/test split provided by dataset

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (recommended) or MPS (Apple Silicon)
- 8GB+ RAM
- 4GB+ GPU memory (for batch_size=64)

## Installation

### Option 1: pip (Recommended)
```bash
pip install -r requirements.txt
```

### Option 2: conda
```bash
conda create -n galaxy-mae python=3.10
conda activate galaxy-mae
pip install -r requirements.txt
```

### Option 3: Docker (Optional)
```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /workspace
WORKDIR /workspace
```

## Advanced Usage

### Custom Model Architecture
Modify the model configuration in `main.py`:

```python
config = {
    "embed_dim": 768,        # Larger model
    "encoder_depth": 12,     # More layers
    "decoder_depth": 8,      # Deeper decoder
    # ... other parameters
}
```

### Custom Dataset
Replace the dataset loading in `data.py`:

```python
# Load your custom dataset
dataset = load_dataset("your-username/your-dataset")
```

### Evaluation Only
```python
from mae_model import create_mae_model
from train_and_eval import evaluate_linear_probe

# Load trained model
model = create_mae_model(**config)
model.load_state_dict(torch.load("path/to/checkpoint.pth"))

# Evaluate
accuracy = evaluate_linear_probe(model.vit, train_loader, test_loader, device)
```

## References

- [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
- [Galaxy10-DECALS Dataset](https://huggingface.co/datasets/matthieulel/galaxy10_decals)

