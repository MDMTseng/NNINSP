import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import glob
import numpy as np
import xml.etree.ElementTree as ET
import random
from pathlib import Path
import logging
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IdentityTransform:
    def __call__(self, x):
        return x

class PlatingDefectDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, cache_size=100):
        """
        初始化數據集
        Args:
            root_dir: 數據根目錄
            transform: 圖像變換
            train: 是否為訓練模式
            cache_size: LRU快取大小
        """
        self.transform = transform
        self.train = train
        self.root_dir = Path(root_dir)
        
        # 設置目錄
        split_dir = "train" if train else "validation"
        self.image_base_dir = self.root_dir / split_dir / "images"
        self.annotation_dir = self.root_dir / split_dir / "annotations"
        
        if not self.image_base_dir.exists():
            raise ValueError(f"Directory not found: {self.image_base_dir}")
        
        # 初始化數據
        self._initialize_dataset()
        
        # 設置數據增強
        # self._setup_transforms()
        
        # 設置快取
        self._setup_cache(cache_size)
        
        logger.info(f"Initialized dataset with {len(self.images)} images in {split_dir} set")
        logger.info(f"Defect type mapping: {self.defect_to_idx}")

    def _initialize_dataset(self):
        """初始化數據集的圖片和標籤"""
        # 獲取所有缺陷類型目錄
        defect_types = ['OK'] + [d.name for d in self.image_base_dir.iterdir() if d.is_dir()]
        
        # 收集圖片和標籤
        self.images = []
        self.labels = []
        self.annotations = []
        
        for defect_type in defect_types[1:]:  # Skip 'OK' when collecting images
            defect_dir = self.image_base_dir / defect_type
            image_paths = list(defect_dir.glob("*.jpg")) + \
                         list(defect_dir.glob("*.bmp")) + \
                         list(defect_dir.glob("*.png"))
            
            self.images.extend(image_paths)
            self.labels.extend([defect_type] * len(image_paths))
            
            # 獲取對應的標註文件
            for img_path in image_paths:
                xml_path = self.annotation_dir / f"{img_path.stem}.xml"
                self.annotations.append(xml_path if xml_path.exists() else None)
        
        # 創建缺陷類型映射 (OK is index 0)
        self.defect_to_idx = {defect: idx for idx, defect in enumerate(defect_types)}
        self.num_classes = len(self.defect_to_idx)

    # def _setup_transforms(self):
    #     """設置數據增強轉換"""
    #     if not self.transform:
    #         return

    #     # 基礎轉換
    #     base_transforms = [
    #         transforms.Resize((256, 256)),
    #         transforms.ToTensor(),
    #     ]

    #     # 訓練時的數據增強
    #     if self.train:
    #         train_transforms = [
    #             transforms.RandomHorizontalFlip(p=0.5),
    #             transforms.RandomVerticalFlip(p=0.5),
    #             transforms.RandomAffine(
    #                 degrees=180,
    #                 translate=(0.1, 0.1),
    #                 scale=(0.8, 1.2),
    #                 fill=0
    #             ),
    #             transforms.ColorJitter(
    #                 brightness=0.2,
    #                 contrast=0.2,
    #                 saturation=0.2
    #             ),
    #             transforms.RandomApply([
    #                 transforms.GaussianBlur(3, sigma=(0.1, 2.0))
    #             ], p=0.5),
    #         ]
    #         base_transforms[1:1] = train_transforms

    #     # 標準化
    #     base_transforms.append(
    #         transforms.Normalize(
    #             mean=[0.485, 0.456, 0.406],
    #             std=[0.229, 0.224, 0.225]
    #         )
    #     )

    #     self.image_transform = transforms.Compose(base_transforms)
        
    #     # 遮罩轉換
    #     self.mask_transform = transforms.Compose([
    #         transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)
    #     ])

    def _setup_cache(self, cache_size):
        """設置LRU快取"""
        @lru_cache(maxsize=cache_size)
        def load_image(path):
            return Image.open(path).convert('RGB')
        
        @lru_cache(maxsize=cache_size)
        def load_annotation(path):
            if path is None:
                return None
            return ET.parse(path).getroot()
        
        self.load_image = load_image
        self.load_annotation = load_annotation

    def create_mask(self, xml_root, image_size, class_idx):
        """從XML創建遮罩"""
        if xml_root is None:
            return torch.zeros((self.num_classes, *image_size))
        
        mask = torch.zeros((self.num_classes, *image_size))
        
        for obj in xml_root.findall('.//object'):
            defect_type = obj.find('name').text
            class_idx = self.defect_to_idx[defect_type]
            bbox = obj.find('bndbox')
            
            coords = [
                int(float(bbox.find(pos).text))
                for pos in ['xmin', 'ymin', 'xmax', 'ymax']
            ]
            
            mask[class_idx, coords[1]:coords[3], coords[0]:coords[2]] = 1.0
            
        return mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        original_size = image.size
        
        # Create multi-channel mask with OK class as default (index 0)
        num_classes = len(self.defect_to_idx)
        mask = torch.zeros((num_classes, original_size[1], original_size[0]))
        mask[0] = 1.0  # Set OK class as default
        
        if idx < len(self.annotations) and self.annotations[idx] is not None:
            tree = ET.parse(self.annotations[idx])
            root = tree.getroot()
            
            for obj in root.findall('.//object'):
                defect_type = obj.find('name').text
                class_idx = self.defect_to_idx[defect_type]
                bbox = obj.find('bndbox')
                
                xmin = max(0, int(float(bbox.find('xmin').text))-5)
                ymin = max(0, int(float(bbox.find('ymin').text))-5)
                xmax = min(original_size[0], int(float(bbox.find('xmax').text))+5)
                ymax = min(original_size[1], int(float(bbox.find('ymax').text))+5)
                
                mask[class_idx, ymin:ymax, xmin:xmax] = 1.0
                mask[0, ymin:ymax, xmin:xmax] = 0.0
        
        if self.train:
            # Convert image to tensor first for faster operations
            image = transforms.ToTensor()(image)
            
            # Add reflection padding efficiently
            padding_size = 150
            padded_image = torch.nn.functional.pad(
                image.unsqueeze(0), 
                (padding_size,)*4, 
                mode='reflect'
            )[0]
            
            # Pad all masks at once
            padded_mask = torch.nn.functional.pad(
                mask, 
                (padding_size,)*4, 
                mode='reflect'
            )
            
            # Apply geometric transforms
            if random.random() < 0.5:
                padded_image = torch.flip(padded_image, [-1])
                padded_mask = torch.flip(padded_mask, [-1])
            
            # Apply affine transform
            affine_params = transforms.RandomAffine.get_params(
                degrees=(-180, 180),
                translate=(0.1, 0.1),
                scale_ranges=(0.9, 1.4),
                shears=None,
                img_size=padded_image.shape[-2:]
            )
            
            padded_image = transforms.functional.affine(
                padded_image, *affine_params, interpolation=transforms.InterpolationMode.BILINEAR
            )
            padded_mask = transforms.functional.affine(
                padded_mask, *affine_params, interpolation=transforms.InterpolationMode.NEAREST
            )
            
            # NEW: Random brightness/exposure adjustment
            if random.random() < 0.7:  # 70% chance to apply
                brightness_factor = random.uniform(0.7, 1.3)
                padded_image = transforms.functional.adjust_brightness(padded_image, brightness_factor)
            
            # NEW: Random contrast adjustment
            if random.random() < 0.5:  # 50% chance to apply
                contrast_factor = random.uniform(0.8, 1.2)
                padded_image = transforms.functional.adjust_contrast(padded_image, contrast_factor)
                
            # NEW: Add gradient tint (simulates uneven lighting)
            if random.random() < 0.4:  # 40% chance to apply
                h, w = padded_image.shape[-2:]
                
                # Create random gradient direction
                direction = random.choice(['horizontal', 'vertical', 'diagonal'])
                strength = random.uniform(0.05, 0.2)  # Controls the intensity of the gradient
                
                # Create gradient mask
                if direction == 'horizontal':
                    gradient = torch.linspace(1.0-strength, 1.0+strength, w).view(1, w).expand(h, w)
                elif direction == 'vertical':
                    gradient = torch.linspace(1.0-strength, 1.0+strength, h).view(h, 1).expand(h, w)
                else:  # diagonal
                    gradient_h = torch.linspace(0, 1, h).view(h, 1).expand(h, w)
                    gradient_w = torch.linspace(0, 1, w).view(1, w).expand(h, w)
                    gradient = 1.0 + strength * (gradient_h + gradient_w - 1.0)
                
                # Apply gradient per channel with random color tint
                for c in range(3):
                    if random.random() < 0.7:  # Apply to some channels only
                        padded_image[c] = padded_image[c] * gradient
            
            # Center crop
            crop_size = 256
            h, w = padded_image.shape[-2:]
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
            
            image = padded_image[:, top:top+crop_size, left:left+crop_size]
            mask = padded_mask[:, top:top+crop_size, left:left+crop_size]
            
            # No need to normalize since SegFormer expects [0,1] range
            
        else:
            # For validation, only apply basic transforms
            image = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])(image)
            
            mask = transforms.Resize(
                (256, 256), 
                interpolation=transforms.InterpolationMode.NEAREST
            )(mask)
        
        # Convert mask to class indices
        mask = mask.argmax(dim=0)
        
        return image, mask, label 

    def get_defect_distribution(self):
        """獲取數據集中缺陷類型的分佈"""
        distribution = {}
        for label in self.labels:
            distribution[label] = distribution.get(label, 0) + 1
        return distribution 

    def get_class_names(self):
        """
        Returns a list of class names indexed by their corresponding class index
        
        Returns:
            list: List of class names where index corresponds to class index
        """
        # Create a list to store class names in order
        class_names = [""] * len(self.defect_to_idx)
        
        # Fill the list using the defect_to_idx mapping
        for defect_name, idx in self.defect_to_idx.items():
            class_names[idx] = defect_name
        
        return class_names

    def get_class_name(self, class_idx):
        """
        Get class name for a specific class index
        
        Args:
            class_idx (int): The class index
            
        Returns:
            str: The class name corresponding to the index, or 'Unknown' if not found
        """
        class_names = self.get_class_names()
        if 0 <= class_idx < len(class_names):
            return class_names[class_idx]
        return "Unknown" 