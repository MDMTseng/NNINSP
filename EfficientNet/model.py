import torch
import torch.nn as nn
import timm

class SegmentationModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # Create encoder using timm
        self.encoder = timm.create_model('efficientnet_b2', 
                                       pretrained=True, 
                                       features_only=True)
        
        base_model = timm.create_model('efficientnet_b2', pretrained=True)
        self.input_size = base_model.default_cfg['input_size']  # Returns (3, 260, 260)
        del base_model  # Clean up the temporary model
        print(f"Input size: {self.input_size}")



        # Get feature channels for each stage
        self.channels = self.encoder.feature_info.channels()
        
        # Create decoder
        # Using the last feature map (highest level features)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.channels[-1], 512, kernel_size=4, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, num_classes, kernel_size=4, stride=1),
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)
        )
    
    def forward(self, x):
        # Get all feature maps from encoder
        features = self.encoder(x)
        # Use the last feature map
        x = features[-1]
        # Apply decoder
        x = self.decoder(x)
        return x 
    

    