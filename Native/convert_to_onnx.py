import torch
import sys
import os
import platform
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from EfficientNet.model import SegmentationModel

def convert_to_onnx(checkpoint_path):
    # 載入模型
    model = SegmentationModel()
    try:
        # 嘗試載入新格式的 checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 如果是舊格式，直接載入
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    print("Model loaded successfully")
    model.eval()
    
    # 生成示例輸入
    dummy_input = torch.randn(1, 3, 256, 256)
    
    # 導出 ONNX
    onnx_path = "model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model converted to ONNX format: {onnx_path}")

    # 如果是 macOS，將 ONNX 轉換為 CoreML
    if platform.system() == 'Darwin':
        import coremltools as ct
        
        # 從 ONNX 載入並轉換為 CoreML
        model = ct.converters.onnx.convert(
            model=onnx_path,
            preprocessing_args={
                'image_input_names': ['input'],
                'image_scale': 1.0/255.0,
                'bias': [-0.485, -0.456, -0.406],
                'scale': [1.0/0.229, 1.0/0.224, 1.0/0.225]
            },
            minimum_deployment_target=ct.target.iOS13
        )
        
        # 設置模型元數據
        model.author = 'Your Name'
        model.license = 'MIT'
        model.short_description = 'Steel Surface Defect Detection'
        
        # 設置輸入格式
        model.input_description['input'] = 'Input image (256x256)'
        model.output_description['output'] = 'Segmentation map'
        
        # 保存模型
        mlmodel_path = "model.mlmodel"
        model.save(mlmodel_path)
        print(f"Model converted to CoreML format: {mlmodel_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX/CoreML')
    parser.add_argument('checkpoint_path', type=str, help='Path to the PyTorch checkpoint file (.pth)')
    args = parser.parse_args()
    
    convert_to_onnx(args.checkpoint_path) 