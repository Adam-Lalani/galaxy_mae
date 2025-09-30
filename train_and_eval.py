import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
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
        if isinstance(model, nn.DataParallel):
            loss = loss.mean()
            
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    """
    print("--- Starting Linear Probe Evaluation ---")
    
    # 1. Freeze the encoder's weights
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    # 2. Create and prepare the linear classifier (the "probe")
    probe = nn.Linear(encoder.config.hidden_size, num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 3. Train the linear probe with a new progress bar
    for epoch in range(probe_epochs):
        probe.train()
        for batch in tqdm(train_loader, desc=f"Probing Epoch {epoch+1}/{probe_epochs}", leave=False):
            images = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            with torch.no_grad():
                features = encoder(pixel_values=images).last_hidden_state
                cls_feature = features[:, 0]
            
            outputs = probe(cls_feature)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 4. Evaluate the trained probe on the test set
    probe.eval()
    correct, total = 0, 0
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
        print(f"Probing Epoch {epoch+1}/{probe_epochs} | Accuracy: {correct/total:.4f}")
    return correct/total


def fine_tune_and_evaluate(
    encoder, 
    train_loader, 
    test_loader, 
    device,
    num_classes=10, 
    finetune_epochs=50,
    lr=1e-5,
    warmup_epochs=5,
    min_lr=1e-6
):
    """
    Fine-tunes the entire encoder and a new classification head jointly
    """
    print(f"\n{'='*50}\n--- Starting Full Fine-tuning for {finetune_epochs} epochs ---\n{'='*50}")
    
    # Create a classification head
    classifier = nn.Linear(encoder.config.hidden_size, num_classes).to(device)
    
    # Enable gradients for both encoder and classifier
    for param in encoder.parameters():
        param.requires_grad = True
    
    # Optimizer for both encoder and classifier
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(classifier.parameters()), 
        lr=lr, 
        weight_decay=0.05
    )
    criterion = nn.CrossEntropyLoss()
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    main_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=finetune_epochs - warmup_epochs, 
        eta_min=min_lr
    )
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, main_scheduler], 
        milestones=[warmup_epochs]
    )

    # Fine-tuning loop
    for epoch in range(finetune_epochs):
        encoder.train()
        classifier.train()
        
        for batch in tqdm(train_loader, desc=f"Fine-tuning Epoch {epoch+1}/{finetune_epochs}", leave=False):
            images, labels = batch['pixel_values'].to(device), batch['label'].to(device)
            
            # Forward pass through encoder (returns BaseModelOutput)
            encoder_output = encoder(pixel_values=images)
            cls_features = encoder_output.last_hidden_state[:, 0]  # Use CLS token
            
            # Forward pass through classifier
            logits = classifier(cls_features)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()

    # Final evaluation
    encoder.eval()
    classifier.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch['pixel_values'].to(device), batch['label'].to(device)
            
            # Forward pass through encoder
            encoder_output = encoder(pixel_values=images)
            cls_features = encoder_output.last_hidden_state[:, 0]  # Use CLS token
            
            # Forward pass through classifier
            logits = classifier(cls_features)
            
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = correct / total
    print(f"Fine-tuning Test Accuracy: {accuracy:.2f}%")
    return accuracy