import torch
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm

def calculate_mean_std(dataset_raw, image_size=256, batch_size=64, num_workers=0):
    
    # grab data and resize if needed
    stat_preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    
    def to_rgb_transform(examples):
        examples["pixel_values"] = [stat_preprocess(image.convert("RGB")) for image in examples["image"]]
        del examples["image"]
        return examples
    
    # Use set_transform instead of map to avoid disk space issues
    dataset_raw.set_transform(to_rgb_transform)
    
    # figure out statistics
    loader = torch.utils.data.DataLoader(dataset_raw, batch_size=batch_size, num_workers=num_workers)
    
    mean = 0.
    std = 0.
    num_samples = 0.
    
    print("Calculating dataset statistics (mean and std)...")
    for batch in tqdm(loader, desc="Calculating Stats"):
        images = batch['pixel_values']
        batch_samples = images.size(0)
        # Reshape to (batch_size, channels, height*width) to calculate stats per channel
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        num_samples += batch_samples

    mean /= num_samples
    std /= num_samples
    
    dataset_raw.set_transform(None)
    
    return mean.tolist(), std.tolist()


def get_dataloaders(batch_size=64, image_size=256, num_workers=0):
    """
    Downloads the Galaxy10 dataset and creates the necessary DataLoaders.
    
    Returns:
        mae_loader: DataLoader for the foundation model, using all images (train + test) without labels.
        probe_train_loader: DataLoader for training the linear probe, using the labeled training set.
        probe_test_loader: DataLoader for evaluating the linear probe, using the labeled test set.
    """
    print("Loading Galaxy10 dataset from Hugging Face...")
    dataset = load_dataset("matthieulel/galaxy10_decals")

    # 1. Calculate the actual mean and std of the training data
    train_mean, train_std = calculate_mean_std(dataset['train'], image_size, batch_size)
    print(f"\nCalculated Mean: {train_mean}")
    print(f"Calculated Std: {train_std}")

    # 2. Create transforms for MAE (no augmentation) and probe (with augmentation)
    # MAE pre-training: NO augmentation for better reconstruction
    mae_preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_mean, std=train_std),
    ])
    
    # Linear probe: NO augmentation (standard practice - measures representation quality)
    probe_preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_mean, std=train_std),
    ])
    
    # Fine-tuning: YES augmentation (helps prevent overfitting during full training)
    finetune_train_preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_mean, std=train_std),
    ])
    
    # Test: no augmentation
    test_preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_mean, std=train_std),
    ])

    def mae_transform(examples):
        examples["pixel_values"] = [mae_preprocess(image.convert("RGB")) for image in examples["image"]]
        del examples["image"]
        return examples
    
    def probe_transform(examples):
        examples["pixel_values"] = [probe_preprocess(image.convert("RGB")) for image in examples["image"]]
        del examples["image"]
        return examples
    
    def finetune_train_transform(examples):
        examples["pixel_values"] = [finetune_train_preprocess(image.convert("RGB")) for image in examples["image"]]
        del examples["image"]
        return examples
    
    def test_transform(examples):
        examples["pixel_values"] = [test_preprocess(image.convert("RGB")) for image in examples["image"]]
        del examples["image"]
        return examples

    # Create separate dataset views with different transforms
    # MAE uses all data without augmentation
    train_ds_mae = dataset['train']
    test_ds_mae = dataset['test']
    train_ds_mae.set_transform(mae_transform)
    test_ds_mae.set_transform(mae_transform)
    
    # Probe uses data WITHOUT augmentation (standard practice)
    train_ds_probe = dataset['train']
    test_ds_probe = dataset['test']
    train_ds_probe.set_transform(probe_transform)
    test_ds_probe.set_transform(test_transform)
    
    # Fine-tuning uses data WITH augmentation
    train_ds_finetune = dataset['train']
    test_ds_finetune = dataset['test']
    train_ds_finetune.set_transform(finetune_train_transform)
    test_ds_finetune.set_transform(test_transform)
    
    # 3. Create combined dataset for MAE (all data, no augmentation)
    full_dataset_for_mae = torch.utils.data.ConcatDataset([train_ds_mae, test_ds_mae])
    
    # 4. Create the DataLoaders
    mae_loader = torch.utils.data.DataLoader(
        full_dataset_for_mae, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    
    probe_train_loader = torch.utils.data.DataLoader(
        train_ds_probe, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    
    probe_test_loader = torch.utils.data.DataLoader(
        test_ds_probe, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    finetune_train_loader = torch.utils.data.DataLoader(
        train_ds_finetune, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    
    finetune_test_loader = torch.utils.data.DataLoader(
        test_ds_finetune, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    print("\nDataloaders created successfully.")
    print(f"  - MAE loader: No augmentation (for reconstruction)")
    print(f"  - Probe loaders: No augmentation (for evaluation)")
    print(f"  - Fine-tune loaders: WITH augmentation (for training)")
    
    return mae_loader, probe_train_loader, probe_test_loader, finetune_train_loader, finetune_test_loader, train_mean, train_std

