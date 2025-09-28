import torch
import torch.nn as nn
from tqdm import tqdm

def train_mae_one_epoch(model, dataloader, optimizer, device):
    """
    Trains the Masked Autoencoder (MAE) model for one epoch.

    Args:
        model (nn.Module): The MAE model to be trained.
        dataloader (DataLoader): DataLoader for the pre-training data.
        optimizer (Optimizer): The optimizer for the model.
        device (str): The device to train on ('cuda' or 'cpu').

    Returns:
        float: The average reconstruction loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="MAE Pre-training Epoch"):
        images = batch['pixel_values'].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(pixel_values=images)
        loss = outputs.loss  # The Hugging Face model calculates the loss internally

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Calculate and return the average loss for the epoch
    return total_loss / len(dataloader)


def evaluate_linear_probe(
    encoder, 
    train_loader, 
    test_loader, 
    device,
    num_classes=10, 
    probe_epochs=10,
    lr=1e-3
):
    """
    Evaluates the quality of the encoder's features using a linear probe.

    Args:
        encoder (nn.Module): The encoder part of the MAE model (e.g., model.vit).
        train_loader (DataLoader): DataLoader for the labeled training data.
        test_loader (DataLoader): DataLoader for the labeled testing data.
        device (str): The device to run on ('cuda' or 'cpu').
        num_classes (int): The number of classes for classification.
        probe_epochs (int): Number of epochs to train the linear probe.
        lr (float): Learning rate for the linear probe's optimizer.

    Returns:
        float: The final test accuracy of the trained linear probe.
    """
    print("--- Starting Linear Probe Evaluation ---")
    
    # 1. Freeze the encoder's weights
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval() # Set encoder to evaluation mode

    # 2. Create and prepare the linear classifier
    probe = nn.Linear(encoder.config.hidden_size, num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 3. Train the linear probe
    for epoch in range(probe_epochs):
        probe.train()
        for batch in train_loader:
            images = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            # Extract features from the frozen encoder
            with torch.no_grad():
                # The encoder's output is a tuple; we want the last_hidden_state
                features = encoder(pixel_values=images).last_hidden_state
                # Extract the [CLS] token feature for classification
                cls_feature = features[:, 0]
            
            # Forward pass through the probe
            outputs = probe(cls_feature)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization for the probe
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 4. Evaluate the trained probe on the test set
    probe.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            images = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            features = encoder(pixel_values=images).last_hidden_state
            cls_feature = features[:, 0]
            
            outputs = probe(cls_feature)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()