import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import requests
import zipfile
from tqdm import tqdm
import glob
import random
import shutil
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
import timm
from dataset import PlatingDefectDataset, IdentityTransform
from model import SegmentationModel

def get_device():
    """Get the best available device for Apple Silicon or fallback to CPU"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon) device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device

def validate(model, val_loader, criterion, device):
    """
    Validate the model on the validation set
    """
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for images, masks, _ in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, masks).item()
    
    return val_loss / len(val_loader)

def train_model(model, train_loader, val_loader, device, num_epochs=50, start_epoch=0):
    class DiceLoss(nn.Module):
        def __init__(self, num_classes=6):
            super(DiceLoss, self).__init__()
            self.num_classes = num_classes

        def forward(self, inputs, targets, smooth=1):
            # Convert inputs to probabilities
            inputs = torch.softmax(inputs, dim=1)
            
            # One-hot encode targets
            targets_one_hot = torch.zeros_like(inputs)
            targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
            
            # Calculate Dice coefficient for each class
            dice = 0
            for i in range(self.num_classes):
                dice += self._dice_coef(inputs[:, i], targets_one_hot[:, i], smooth)
            
            # Return mean Dice loss
            return 1.0 - (dice / self.num_classes)
        
        def _dice_coef(self, inputs, targets, smooth=1):
            intersection = (inputs * targets).sum()
            union = inputs.sum() + targets.sum()
            dice = (2.0 * intersection + smooth) / (union + smooth)
            return dice

    # Combined loss with adjusted weighting (more weight on CrossEntropy)
    criterion = lambda outputs, targets: (
        0.7 * nn.CrossEntropyLoss()(outputs, targets) + 
        0.3 * DiceLoss(num_classes=6)(outputs, targets)
    )

    # Use AdamW with adjusted parameters
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0001,  # Lower learning rate
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.04  # Increased weight decay
    )
    
    # Learning rate scheduler with adjusted parameters
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.2,  # Larger reduction
        patience=3,   # Reduced patience
        verbose=True,
        min_lr=1e-6  # Minimum learning rate
    )

    # Load training history if exists
    history_path = 'training_history.json'
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
            best_val_loss = history['best_val_loss']
            last_epoch = history['last_epoch']
            val_losses = history['val_losses']
            train_losses = history['train_losses']
            print(f"Resuming training from epoch {last_epoch+1}")
            print(f"Best validation loss so far: {best_val_loss:.4f}")
            start_epoch = last_epoch + 1
    else:
        best_val_loss = float('inf')
        val_losses = []
        train_losses = []
    
    # Check if we should reload the best model
    best_model_path = 'best_model.pth'
    if os.path.exists(best_model_path) and len(val_losses) > 0:
        if val_losses[-1] > best_val_loss:
            print("Loading previous best model as current validation score is worse")
            model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for images, masks, _ in progress_bar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            if device.type == "mps":
                torch.mps.synchronize()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'training_loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
            
        # Validation phase
        avg_val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(avg_val_loss)
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print(f'Average Training Loss: {avg_train_loss:.4f}')
        print(f'Average Validation Loss: {avg_val_loss:.4f}')
        
        # Save model if validation loss improved
        if avg_val_loss < best_val_loss or True:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("New best model saved!")
            
            # Save checkpoint with timestamp
            avg_val_loss_str = f'{avg_val_loss:.4f}'
            #replace . with _
            avg_val_loss_str = avg_val_loss_str.replace('.', '_')
            checkpoint_path = f'checkpoint_epoch{epoch+1}_{avg_val_loss_str}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
        else:
            print(f"Validation loss did not improve from {best_val_loss:.4f}")
        
        # Save training history
        history = {
            'best_val_loss': best_val_loss,
            'last_epoch': epoch,
            'val_losses': val_losses,
            'train_losses': train_losses
        }
        with open(history_path, 'w') as f:
            json.dump(history, f)

    return model

def main():
    device = get_device()
    
    # Create transforms for images (without padding - it's handled in dataset)
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets using train and val splits
    train_dataset = PlatingDefectDataset(
        root_dir='NEU-DET',
        transform=image_transform,
        train=True
    )
    
    # Visualize some random samples before training
    print("\nSaving sample visualizations...")
    num_samples = 0
    sample_indices = random.sample(range(len(train_dataset)), num_samples)
    
    # Create output directory
    os.makedirs('sample_visualizations', exist_ok=True)
    
    for idx in sample_indices:
        image, mask, label = train_dataset[idx]
        
        # Convert image tensor back to displayable image
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_display = image * std + mean
        image_display = (image_display * 255).clamp(0, 255).numpy().transpose(1, 2, 0).astype(np.uint8)
        
        # Create figure
        plt.figure(figsize=(20, 10))
        
        # Plot original image
        plt.subplot(2, 4, 1)
        plt.imshow(image_display)
        plt.title(f'Input Image\nDefect: {label}')
        plt.axis('off')
        
        # Plot ground truth mask for each class
        defect_types = train_dataset.defect_to_idx.keys()
        for i, defect_type in enumerate(defect_types):
            plt.subplot(2, 4, i + 2)
            mask_idx = train_dataset.defect_to_idx[defect_type]
            plt.imshow(mask == mask_idx, cmap='gray')
            plt.title(f'{defect_type} Mask')
            plt.axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f'sample_visualizations/sample_{idx}.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Saved visualization for sample {idx}")
    
    print("Sample visualizations saved to 'sample_visualizations' directory")
    if num_samples>0:
        exit()
    val_dataset = PlatingDefectDataset(
        root_dir='NEU-DET',
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ]),
        train=False
    )
    
    # Create dataloaders with settings for M1
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32,
        num_workers=0,
        pin_memory=True
    )
    
    # Initialize model
    model = SegmentationModel()
    
    # Load saved model if exists
    saved_model_path = 'best_model.pth'  # or 'final_model.pth'
    if os.path.exists(saved_model_path):
        print(f"Loading saved model from {saved_model_path}")
        model.load_state_dict(torch.load(saved_model_path, map_location=device))
    else:
        print("Starting with fresh model")
    
    model = model.to(device)
    
    # Train model
    trained_model = train_model(model, train_loader, val_loader, device)
    
    # Save the final model with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'model_{timestamp}.pth'
    torch.save(trained_model.state_dict(), save_path)
    print(f"Training completed! Model saved to {save_path}")

if __name__ == '__main__':
    main() 