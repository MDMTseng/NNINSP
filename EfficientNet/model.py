import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from torch.serialization import safe_globals
import numpy as np
import torch.nn.functional as F

class SegmentationModel(nn.Module):
    def __init__(self, num_classes=7, model_variant=2):
        super().__init__()
        

        # Configure SegFormer (choose from B0 to B4 with index 0-4)
        self.model_variant = model_variant  # Default to B0, can be configured to 4 for B4
        model_config = self.get_segFormer_config(self.model_variant, num_classes)
        self.config = SegformerConfig(**model_config)
        
        # Create SegFormer model
        self.segformer = SegformerForSemanticSegmentation(self.config)
        
        # Add upsampling layer to match target size
        self.upsample = nn.Upsample(
            size=(256, 256), 
            mode='bilinear', 
            align_corners=False
        )
        
        print(f"Initialized SegFormer with {num_classes} classes")
        print(f"Input size: (3, 256, 256)")

    def get_segFormer_config(self, model_variant, num_classes):
        configs = [
        {  # b0 (smallest)
            "num_channels": 3,
            "num_encoder_blocks": 4,
            "depths": [2, 2, 2, 2],
            "sr_ratios": [8, 4, 2, 1],
            "hidden_sizes": [32, 64, 160, 256],
            "decoder_hidden_size": 256,
            "num_labels": num_classes,
            "image_size": 256
        },
        {  # b1
            "num_channels": 3,
            "num_encoder_blocks": 4,
            "depths": [2, 2, 2, 2],
            "sr_ratios": [8, 4, 2, 1],
            "hidden_sizes": [64, 128, 320, 512],
            "decoder_hidden_size": 256,
            "num_labels": num_classes,
            "image_size": 256
        },
        {  # b2
            "num_channels": 3,
            "num_encoder_blocks": 4,
            "depths": [3, 4, 6, 3],
            "sr_ratios": [8, 4, 2, 1],
            "hidden_sizes": [64, 128, 320, 512],
            "decoder_hidden_size": 768,
            "num_labels": num_classes,
            "image_size": 256
        },
        {  # b3
            "num_channels": 3,
            "num_encoder_blocks": 4,
            "depths": [3, 4, 18, 3],
            "sr_ratios": [8, 4, 2, 1],
            "hidden_sizes": [64, 128, 320, 512],
            "decoder_hidden_size": 768,
            "num_labels": num_classes,
            "image_size": 256
        },
        {  # b4 (largest)
            "num_channels": 3,
            "num_encoder_blocks": 4,
            "depths": [3, 8, 27, 3],
            "sr_ratios": [8, 4, 2, 1],
            "hidden_sizes": [64, 128, 320, 512],
            "decoder_hidden_size": 768,
            "num_labels": num_classes,
            "image_size": 256
        }
        ]
        
        return configs[model_variant]
    def get_segFormer_recommend_training_config(self, model_variant):
        tconf = [
        { #b0
            "lr": 0.0001,  # Higher learning rate for smaller model
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.01,  # Standard transformer regularization
            "batch_size": 32,      # Can use larger batches for smaller model
            "epochs": 100,         # Smaller model may need more epochs
            "scheduler": "cosine"  # Cosine annealing recommended
        },
        { #b1
            "lr": 0.00006,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.01,
            "batch_size": 24,
            "epochs": 80,
            "scheduler": "cosine"
        },
        { #b2
            "lr": 0.00006,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.01,
            "batch_size": 16,
            "epochs": 80,
            "scheduler": "cosine"
        },
        { #b3
            "lr": 0.00005,  # Slightly reduced learning rate for larger model
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.01,
            "batch_size": 8,     # Smaller batch size due to memory requirements
            "epochs": 60,        # Larger models often need fewer epochs
            "scheduler": "cosine"
        },
        { #b4
            "lr": 0.00004,  # Even lower learning rate for the largest model
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.01,
            "batch_size": 4,     # Very small batch size due to memory requirements
            "epochs": 50,        # Even fewer epochs for the largest model
            "scheduler": "cosine"
        }]
        return tconf[model_variant]

    def forward(self, x):
        """
        Forward pass with device compatibility handling
        """
        # Ensure input is on the same device as model weights
        device = next(self.parameters()).device
        x = x.to(device)
        
        # Forward pass through SegFormer
        outputs = self.segformer(pixel_values=x)
        
        # Upsample logits to match target size
        logits = self.upsample(outputs.logits)
        
        return logits
    
    def save_model(self, path):
        """Save both model architecture and weights"""
        save_dict = {
            'config': self.config,
            'state_dict': self.state_dict(),
            'segformer_state': self.segformer.state_dict()
        }
        torch.save(save_dict, path)

    @classmethod
    def load_model(cls, path, device=None):
        """Load both model architecture and weights"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Allow SegformerConfig to be loaded
        from transformers.models.segformer.configuration_segformer import SegformerConfig
        torch.serialization.add_safe_globals([SegformerConfig])
        
        save_dict = torch.load(path, map_location=device, weights_only=False)
        
        # Create new model with saved config
        model = cls(num_classes=save_dict['config'].num_labels)
        
        # Load states
        model.load_state_dict(save_dict['state_dict'])
        model.segformer.load_state_dict(save_dict['segformer_state'])
        
        return model.to(device)
    

class CRFPostProcessor:
    def __init__(self, num_classes=7):
        self.num_classes = num_classes
        
    def process(self, image, logits):
        """Apply CRF post-processing to refine segmentation
        
        Args:
            image: Original image tensor (1, C, H, W)
            logits: Model prediction logits (1, num_classes, H, W)
            
        Returns:
            Refined segmentation mask
        """
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_softmax
        except ImportError:
            print("pydensecrf not installed. Run: pip install pydensecrf")
            return logits.argmax(1)
            
        # Convert to numpy
        img = image[0].permute(1, 2, 0).cpu().numpy()
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        
        # CRF parameters
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], self.num_classes)
        U = unary_from_softmax(probs)
        d.setUnaryEnergy(U)
        
        # Add pairwise potentials
        d.addPairwiseGaussian(sxy=3, compat=3)  # Smooth segmentation
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=img, compat=10)  # Appearance-based
        
        # Inference
        Q = d.inference(5)  # 5 iterations
        map_soln = np.argmax(np.array(Q).reshape(self.num_classes, img.shape[0], img.shape[1]), axis=0)
        
        return torch.tensor(map_soln).unsqueeze(0)
    

    