import torch
import sys
import os
import platform
import argparse
import coremltools as ct

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from EfficientNet.model import SegmentationModel

def convert_to_coreml(checkpoint_path):
    # 載入模型
    model = SegmentationModel()
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    print("Model loaded successfully")
    model.eval()
    
    # 直接轉換到 CoreML
    traced_model = torch.jit.trace(model, torch.randn(1, 3, 256, 256))
    
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.ImageType(
            name="input",
            shape=(1, 3, 256, 256),
            scale=1.0/255.0,
            bias=[-0.485, -0.456, -0.406],
            color_layout="RGB"
        )],
        minimum_deployment_target=ct.target.iOS15,
        convert_to="mlprogram"
    )
    
    # 設置模型元數據
    mlmodel.author = 'Your Name'
    mlmodel.license = 'MIT'
    mlmodel.short_description = 'Steel Surface Defect Detection'
    
    # 保存為 .mlpackage 格式
    mlmodel_path = "model.mlpackage"
    mlmodel.save(mlmodel_path)
    print(f"Model converted to CoreML format: {mlmodel_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert PyTorch model to CoreML')
    parser.add_argument('checkpoint_path', type=str, help='Path to the PyTorch checkpoint file (.pth)')
    args = parser.parse_args()
    
    convert_to_coreml(args.checkpoint_path) 