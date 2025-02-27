import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import requests
import zipfile
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
import glob
import random
import shutil
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime
import json

class IdentityTransform:
    def __call__(self, x):
        return x

class PlatingDefectDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.transform = transform
        self.train = train
        self.root_dir = root_dir
        
        # Get all image paths
        self.images = []
        self.labels = []  # Store defect type labels
        
        # Debug: Print directory structure
        print(f"Looking for images in: {root_dir}")
        
        # Assuming structure: NEU-DET/train/images/{defect_type}/*.jpg
        split_dir = "train" #if train else "val"
        image_base_dir = os.path.join(root_dir, split_dir, "images")
        annotation_dir = os.path.join(root_dir, split_dir, "annotations")
        
        if not os.path.exists(image_base_dir):
            raise ValueError(f"Directory not found: {image_base_dir}")
        
        # Get all defect type folders
        defect_types = [d for d in os.listdir(image_base_dir) 
                       if os.path.isdir(os.path.join(image_base_dir, d))]
        print(f"Found defect types: {defect_types}")
        
        # Collect images from each defect type folder
        for defect_type in defect_types:
            defect_dir = os.path.join(image_base_dir, defect_type)
            image_paths = glob.glob(os.path.join(defect_dir, "*.jpg")) + \
                         glob.glob(os.path.join(defect_dir, "*.bmp")) + \
                         glob.glob(os.path.join(defect_dir, "*.png"))
            
            print(f"Found {len(image_paths)} images in {defect_type} category")
            
            self.images.extend(image_paths)
            self.labels.extend([defect_type] * len(image_paths))
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in {image_base_dir}")
            
        # Sort images to ensure consistent ordering
        # Sort both images and labels together
        sorted_pairs = sorted(zip(self.images, self.labels))
        self.images, self.labels = zip(*sorted_pairs)
        
        # Store annotation paths
        self.annotations = []
        for img_path in self.images:
            # Get corresponding XML file
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            xml_path = os.path.join(annotation_dir, f"{img_name}.xml")
            if os.path.exists(xml_path):
                self.annotations.append(xml_path)
            else:
                print(f"Warning: No annotation found for {img_name}")
        
        print(f"Total images in {split_dir} set: {len(self.images)}")
        
        # Separate transforms for image and mask
        self.image_transform = transform
        
        # Create mask transforms with reflection padding
        mask_transforms = []
        mask_transforms.append(transforms.Resize((256, 256), 
                             interpolation=transforms.InterpolationMode.NEAREST))
        
        if train:
            # Add padding for rotation and translation
            padding_size = 50  # Increased padding for both rotation and translation
            mask_transforms.extend([
                transforms.Pad(padding_size, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(
                    degrees=15,
                    translate=(0.2, 0.2),
                    scale=(0.8, 1.2),
                    fill=None  # None will use reflection padding
                ),
                transforms.CenterCrop(256)  # Crop back to desired size
            ])
        
        self.mask_transform = transforms.Compose(mask_transforms)
        
        # Create mapping of defect types to indices
        self.defect_to_idx = {defect: idx for idx, defect in enumerate(sorted(set(defect_types)))}
        print("Defect type mapping:", self.defect_to_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        original_size = image.size
        
        # Create multi-channel mask
        num_classes = len(self.defect_to_idx)
        mask = torch.zeros((num_classes, original_size[1], original_size[0]))
        
        if idx < len(self.annotations):
            tree = ET.parse(self.annotations[idx])
            root = tree.getroot()
            
            for obj in root.findall('.//object'):
                defect_type = obj.find('name').text
                class_idx = self.defect_to_idx[defect_type]
                bbox = obj.find('bndbox')
                
                xmin = int(float(bbox.find('xmin').text))
                ymin = int(float(bbox.find('ymin').text))
                xmax = int(float(bbox.find('xmax').text))
                ymax = int(float(bbox.find('ymax').text))
                
                mask[class_idx, ymin:ymax, xmin:xmax] = 1.0
        
        # Convert mask to PIL Images for transformations
        mask_pil = [Image.fromarray((m.numpy() * 255).astype(np.uint8)) for m in mask]
        
        # Generate random seed for consistent transformations
        seed = torch.randint(0, 2**32, (1,))[0].item()
        
        if self.image_transform is not None:
            # Add reflection padding to image
            padding_size = 50  # Match the padding size used in mask transforms
            padded_image = transforms.Pad(padding_size, padding_mode='reflect')(image)
            
            # Apply transforms with same seed
            torch.manual_seed(seed)
            random.seed(seed)
            
            # Create custom transform for image that includes padding
            image_transforms = transforms.Compose([
                transforms.Resize((256 + 2*padding_size, 256 + 2*padding_size)),
                transforms.RandomHorizontalFlip(),
                
                transforms.Pad(10, padding_mode='reflect'),  # Apply reflection padding
                transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),

                transforms.CenterCrop(256),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
            
            image = image_transforms(padded_image)
            
            # Apply same geometric transforms to masks
            transformed_masks = []
            for m in mask_pil:
                torch.manual_seed(seed)
                random.seed(seed)
                m = self.mask_transform(m)
                transformed_masks.append(torch.from_numpy(np.array(m)).float() / 255.0)
            
            mask = torch.stack(transformed_masks)
        
        # Convert one-hot encoded mask to class indices
        mask = mask.argmax(dim=0)
        
        return image, mask, label

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

def train_model(model, train_loader, val_loader, device, num_epochs=50, start_epoch=0):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
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
            masks = masks.to(device, non_blocking=True)  # masks shape: [B, H, W]
            
            optimizer.zero_grad()
            outputs = model(images)  # outputs shape: [B, C, H, W]
            
            loss = criterion(outputs, masks)  # CrossEntropyLoss expects [B, C, H, W] and [B, H, W]
            loss.backward()
            optimizer.step()
            
            if device.type == "mps":
                torch.mps.synchronize()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'training_loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
            
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print(f'Average Training Loss: {avg_train_loss:.4f}')
        print(f'Average Validation Loss: {avg_val_loss:.4f}')
        
        # Save model if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("New best model saved!")
            
            # Save checkpoint with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_path = f'checkpoint_epoch{epoch+1}_{timestamp}.pth'
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

class SegmentationModel(nn.Module):
    def __init__(self, num_classes=6):  # 6 defect types
        super().__init__()
        # Load EfficientNet as backbone
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        
        # Remove the last layer
        self.backbone._fc = nn.Identity()
        
        # Add segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(1280, 1280, 1),
            nn.ReLU(),
            nn.Conv2d(1280, 1280, 1),
            nn.ReLU(),
            nn.Conv2d(1280, num_classes, 1),  # Output channel for each defect type
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)
        )
        
    def forward(self, x):
        features = self.backbone.extract_features(x)
        return self.seg_head(features)

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