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
                
                xmin = int(float(bbox.find('xmin').text))-5
                ymin = int(float(bbox.find('ymin').text))-5
                xmax = int(float(bbox.find('xmax').text))+5
                ymax = int(float(bbox.find('ymax').text))+5
                
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(original_size[0], xmax)
                ymax = min(original_size[1], ymax)
                
                # Set defect area and remove OK class in that area
                mask[class_idx, ymin:ymax, xmin:xmax] = 1.0
                mask[0, ymin:ymax, xmin:xmax] = 0.0
        
        # if self.image_transform is not None:
        if True:
            # Add reflection padding to both image and mask
            padding_size = 150
            padded_image = transforms.Pad(padding_size, padding_mode='reflect')(image)
            padded_masks = [transforms.Pad(padding_size, padding_mode='reflect')(Image.fromarray((m.numpy() * 255).astype(np.uint8))) 
                        for m in mask]
            
            # Generate random seed for consistent transformations
            seed = torch.randint(0, 2**32, (1,))[0].item()
            
            # Create transform sequence
            aug_transforms = transforms.Compose([
                # transforms.Resize((256 + 2*padding_size, 256 + 2*padding_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(
                    degrees=20,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.4),
                    fill=None  # Use reflection padding
                ),
                transforms.CenterCrop(256),
            ])
            
            # Apply same geometric transforms to both image and masks
            torch.manual_seed(seed)
            random.seed(seed)
            image = aug_transforms(padded_image)
            
            transformed_masks = []
            for m in padded_masks:
                torch.manual_seed(seed)
                random.seed(seed)
                transformed_mask = aug_transforms(m)
                transformed_masks.append(torch.from_numpy(np.array(transformed_mask)).float() / 255.0)
            
            # Apply color transforms only to image
            color_transforms = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ])
            image = color_transforms(image)
            
            # Stack transformed masks
            mask = torch.stack(transformed_masks)
        
        # Convert one-hot encoded mask to class indices
        mask = mask.argmax(dim=0)
        
        return image, mask, label 

    def get_defect_distribution(self):
        """獲取數據集中缺陷類型的分佈"""
        distribution = {}
        for label in self.labels:
            distribution[label] = distribution.get(label, 0) + 1
        return distribution 