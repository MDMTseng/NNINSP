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
import argparse
from PIL import Image
import torch.nn.functional as F
from dataset import PlatingDefectDataset, IdentityTransform
from model import SegmentationModel

# Function to save augmented images and masks for debugging
def save_augmented_samples(images, masks, filenames, save_dir, count, class_names):
    """
    Save augmented training images and their masks for debugging purposes.
    
    Args:
        images: Batch of images (B, C, H, W)
        masks: Batch of masks (B, H, W)
        filenames: List of original filenames
        save_dir: Directory to save images to
        count: Number of samples to save
        class_names: List of class names from the dataset
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Ensure we don't try to save more images than we have
    count = min(count, len(images))
    
    # Select random indices if we have more images than needed
    indices = list(range(len(images)))
    random.shuffle(indices)
    indices = indices[:count]
    
    # Color map for different classes (dynamic based on number of classes)
    num_classes = len(class_names)
    colors = [
        [0, 0, 0],       # Background - black
        [255, 0, 0],     # Class 1 - red
        [0, 255, 0],     # Class 2 - green
        [0, 0, 255],     # Class 3 - blue
        [255, 255, 0],   # Class 4 - yellow
        [255, 0, 255],   # Class 5 - magenta
        [0, 255, 255],   # Class 6 - cyan
    ]
    
    # Add more colors if needed
    while len(colors) < num_classes:
        colors.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
    
    for i, idx in enumerate(indices):
        # Get the image, mask and filename
        img = images[idx].clone().cpu()
        mask = masks[idx].clone().cpu()
        basename = os.path.splitext(os.path.basename(filenames[idx]))[0]
        
        # Convert tensor to numpy for visualization
        img_np = img.permute(1, 2, 0).numpy()  # Change from C,H,W to H,W,C
        
        # Convert mask to color image for visualization
        mask_np = mask.numpy()
        colored_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
        
        # Apply colors to mask
        for class_idx, color in enumerate(colors):
            colored_mask[mask_np == class_idx] = color
            
        # Create a figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot original image
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Plot colored mask
        axes[1].imshow(colored_mask)
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')
        
        # Plot overlay 
        overlay = img_np.copy()
        mask_overlay = colored_mask.astype(float) / 255
        overlay = overlay * 0.7 + mask_overlay * 0.3  # Blend with 30% opacity
        overlay = np.clip(overlay, 0, 1)
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        # Add legend
        patches = [plt.Rectangle((0, 0), 1, 1, color=[c/255 for c in colors[i]]) 
                  for i in range(len(class_names))]
        plt.figlegend(patches, class_names, loc='lower center', ncol=min(7, len(class_names)), 
                     bbox_to_anchor=(0.5, -0.05))
        
        # Adjust layout to make room for the legend
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.suptitle(f"Sample: {basename}", fontsize=16)
        
        # Save the combined visualization
        viz_path = os.path.join(save_dir, f"{basename}_aug_{i}_viz.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved augmented sample {i+1}/{count}: {viz_path}")

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
    
    # Add progress bar for validation
    progress_bar = tqdm(val_loader, desc=f'Validation')
    
    with torch.no_grad():
        for images, masks, _ in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            batch_loss = criterion(outputs, masks).item()
            val_loss += batch_loss
            
            # Update progress bar with current validation loss
            progress_bar.set_postfix({'val_loss': f'{batch_loss:.4f}'})
    
    return val_loss / len(val_loader)

def detailed_validate(model, val_loader, criterion, device, epoch, class_names):
    # ... existing code ...
    
    # Print metrics using class names
    print("\nPer-class validation accuracy:")
    for c in range(len(class_names)):
        if class_total[c] > 0:
            accuracy = 100 * class_correct[c] / class_total[c]
            print(f"{class_names[c]}: {accuracy:.2f}%")
    
    # Log problematic samples with class names
    print(f"\nFound {len(problematic_samples)} samples with whole-image misclassification")
    if len(problematic_samples) > 0:
        for ps in problematic_samples:
            # Add human-readable class names
            ps['pred_class_name'] = class_names[ps['pred_class']]
            ps['true_class_name'] = class_names[ps['true_class']]
        
        with open('problematic_samples.json', 'w') as f:
            json.dump(problematic_samples, f)

def train_model(model, train_loader, val_loader, device, class_names, num_epochs=300, start_epoch=0, aug_debug=0, model_variant=0):
    # Improved Dice Loss
    class DiceLoss(nn.Module):
        def __init__(self, num_classes=None, smooth=1.0):
            super(DiceLoss, self).__init__()
            self.num_classes = num_classes or len(class_names)
            self.smooth = smooth

        def forward(self, inputs, targets):
            # Convert to probability distribution
            inputs = torch.softmax(inputs, dim=1)

            # One-hot encode targets (ensure targets are long)
            targets_one_hot = torch.zeros_like(inputs)
            targets_one_hot.scatter_(1, targets.unsqueeze(1).long(), 1)

            # Calculate Dice coefficient for each class
            dice_loss = 0
            for i in range(self.num_classes):
                dice_loss += self._dice_coef(inputs[:, i], targets_one_hot[:, i])
            
            return 1.0 - (dice_loss / self.num_classes)
        
        def _dice_coef(self, inputs, targets):
            intersection = (inputs * targets).sum()
            union = inputs.sum() + targets.sum()
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            return dice

    # Focal Loss (replacing CrossEntropyLoss for better handling of class imbalance)
    class FocalLoss(nn.Module):
        def __init__(self, alpha=0.25, gamma=2, num_classes=None):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.num_classes = num_classes or len(class_names)

        def forward(self, inputs, targets):
            log_probs = F.log_softmax(inputs, dim=1)
            probs = torch.exp(log_probs)
            targets_one_hot = torch.zeros_like(inputs)
            targets_one_hot.scatter_(1, targets.unsqueeze(1).long(), 1)

            focal_loss = -self.alpha * (1 - probs) ** self.gamma * targets_one_hot * log_probs
            return focal_loss.sum(dim=1).mean()

    # Combined Loss (Focal + Dice)
    class CombinedLoss(nn.Module):
        def __init__(self, num_classes=None, focal_weight=0.8, dice_weight=0.2):
            super(CombinedLoss, self).__init__()
            self.focal_weight = focal_weight
            self.dice_weight = dice_weight
            self.focal_loss = FocalLoss(num_classes=num_classes)
            self.dice_loss = DiceLoss(num_classes=num_classes)

        def forward(self, outputs, targets):
            return 3* (
                self.focal_weight * self.focal_loss(outputs, targets) + 
                self.dice_weight * self.dice_loss(outputs, targets)
            )

    # Initialize the combined loss with dynamic number of classes
    criterion = CombinedLoss(num_classes=len(class_names))

    # Get training config from the model
    training_config = model.get_segFormer_recommend_training_config(model_variant)
    
    # Optimized optimizer settings using config
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config['lr'],
        betas=training_config['betas'],
        eps=training_config['eps'],
        weight_decay=training_config['weight_decay']
    )
    
    # Learning rate scheduler - polynomial decay works best for SegFormer
    scheduler = optim.lr_scheduler.PolynomialLR(
        optimizer,
        total_iters=len(train_loader) * num_epochs,
        power=1.0  # Linear decay
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
        
        # For augmentation debugging - only save in the first epoch
        aug_samples_saved = 0
        
        for images, masks, filenames in progress_bar:
            # Save augmented samples if requested and we haven't saved enough yet
            # if epoch == start_epoch and aug_debug > 0 and aug_samples_saved < aug_debug:
            #     save_count = min(aug_debug - aug_samples_saved, len(images))
            #     save_augmented_samples(
            #         images, masks, filenames, 
            #         save_dir='aug_debug_samples', 
            #         count=save_count
            #     )
            #     aug_samples_saved += save_count
            
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
        scheduler.step()
        
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print(f'Average Training Loss: {avg_train_loss:.4f}')
        print(f'Average Validation Loss: {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss or True:
            best_val_loss = avg_val_loss
            model.save_model('best_model.pth')
            print("New best model saved!")
        # Save best model if validation loss improved
        
        # model.save_model('latest_model.pth')
        # print("New best model saved!")
    
        # Save periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or os.path.exists(".SAVE_CHECKPOINT"):
            avg_val_loss_str = f'{avg_val_loss:.4f}'.replace('.', '_')
            periodic_checkpoint_path = f'checkpoint_epoch{epoch+1}_{avg_val_loss_str}.pth'
            model.save_model(periodic_checkpoint_path)
            print(f"Periodic checkpoint saved at epoch {epoch+1}")

            # Check for user-requested checkpoint save
            if os.path.exists(".SAVE_CHECKPOINT"):
                # Delete the trigger file
                try:
                    os.remove(".SAVE_CHECKPOINT")
                    print("Checkpoint trigger file removed.")
                except Exception as e:
                    print(f"Warning: Could not remove checkpoint trigger file: {e}")
            



        # # Save periodic checkpoint every 10 epochs
        # if (epoch + 1) % 10 == 0:
        #     periodic_checkpoint_path = f'checkpoint_epoch{epoch+1}.pth'
        #     model.save_model(periodic_checkpoint_path)
        #     print(f"Periodic checkpoint saved at epoch {epoch+1}")
        # else:
        #     print(f"Validation loss did not improve from {best_val_loss:.4f}")
        
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
    # Add command-line argument parsing
    parser = argparse.ArgumentParser(description="Train plating defect detection model")
    parser.add_argument('--aug_dbg', type=int, default=0, 
                        help='Number of augmented training samples to save for debugging')
    # parser.add_argument('--model_variant', type=int, default=2, choices=[0, 1, 2, 3, 4],
    #                    help='SegFormer model variant (0=B0, 1=B1, 2=B2, 3=B3, 4=B4)')
    args = parser.parse_args()
    
    device = get_device()
    
    # Create transforms for images (without normalization for SegFormer)
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Create datasets
    train_dataset = PlatingDefectDataset(
        root_dir='NEU-DET',
        transform=image_transform,
        train=True
    )
    
    # Get class names from dataset
    class_names = train_dataset.get_class_names()
    print(f"Detected classes: {class_names}")
    
    val_dataset = PlatingDefectDataset(
        root_dir='NEU-DET',
        transform=image_transform,
        train=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=16,
        num_workers=0,
        pin_memory=True
    )
    
    # Debug augmented data if requested
    if args.aug_dbg > 0:
        print(f"Saving {args.aug_dbg} augmented samples for debugging...")
        # Get one batch from the train loader
        images, masks, filenames = next(iter(train_loader))
        
        # Save the augmented samples with class names from dataset
        save_augmented_samples(
            images, masks, filenames,
            save_dir='aug_debug_samples',
            count=args.aug_dbg,
            class_names=class_names
        )
        
        print("Augmentation debugging completed. Exiting...")
        return  # Exit the function early
    
    # Pass class names to train_model 
    # Initialize model
    model = SegmentationModel(num_classes=len(class_names))
    
    # Load saved model if exists
    saved_model_path = 'best_model.pth'
    if os.path.exists(saved_model_path):
        print(f"Loading saved model from {saved_model_path}")
        model = SegmentationModel.load_model(saved_model_path, device)
    else:
        print("Starting with fresh model")
        model = SegmentationModel().to(device)
    
    # Store the model variant with the model
    # model.model_variant = model.model_variant
    
    # Pass model variant to train_model
    trained_model = train_model(model, train_loader, val_loader, device, 
                              class_names=class_names, aug_debug=args.aug_dbg,
                              model_variant=model.model_variant)
    
    # Save the final model with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'model_{timestamp}.pth'
    torch.save(trained_model.state_dict(), save_path)
    print(f"Training completed! Model saved to {save_path}")

if __name__ == '__main__':
    main() 