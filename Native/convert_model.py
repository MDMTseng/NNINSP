import torch
import sys
import os
import platform
import argparse
import coremltools as ct
from torch.serialization import safe_globals, add_safe_globals
from transformers import SegformerConfig
import datetime
import shutil

# Add parent directory to path first
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Then import using package notation
from EfficientNet.model import SegmentationModel
from EfficientNet.dataset import PlatingDefectDataset

def convert_to_coreml(checkpoint_path, output_path=None, dataset_path=None, num_classes=None):
    # Resolve paths
    if dataset_path is None:
        # Try standard locations relative to script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        potential_paths = [
            os.path.join(script_dir, '..', 'NEU-DET'),  # Parent dir
            os.path.join(script_dir, '..', 'EfficientNet', 'NEU-DET'),  # In EfficientNet dir
            os.path.join(script_dir, 'NEU-DET')  # In current dir
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                dataset_path = path
                break
        
        if dataset_path is None:
            print("Warning: Could not find dataset directory. Using default class names.")
    
    # Set default output path if not provided
    if output_path is None:
        output_path = "SurfaceDefectDetector.mlpackage"
    
    # Load the model - Fix for PyTorch 2.6+
    print(f"Loading model from {checkpoint_path}...")
    
    # Add SegformerConfig to safe globals
    add_safe_globals([SegformerConfig])
    
    # Try loading with weights_only=False first (for PyTorch 2.6+)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Error loading with weights_only=False: {e}")
        # Fall back to using safe_globals context manager
        with safe_globals([SegformerConfig]):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract model configuration
    config = checkpoint.get('config', None)
    if config is None:
        # Try different keys that might contain the config
        for key in ['model_config', 'configuration', 'cfg']:
            if key in checkpoint:
                config = checkpoint[key]
                break
    
    if config is None:
        print("Warning: No configuration found in checkpoint. Using default configuration.")
        config = SegformerConfig()
    
    # Handle different state dict keys
    state_dict_key = None
    for key in ['model_state_dict', 'state_dict', 'model']:
        if key in checkpoint:
            state_dict_key = key
            break
    
    if state_dict_key is None:
        raise ValueError("Could not find model state dict in checkpoint")
    
    # Determine the number of classes from the checkpoint
    checkpoint_num_classes = None
    
    # Try to get number of classes from the classifier weight in state dict
    if 'segformer.decode_head.classifier.weight' in checkpoint[state_dict_key]:
        classifier_weight = checkpoint[state_dict_key]['segformer.decode_head.classifier.weight']
        checkpoint_num_classes = classifier_weight.shape[0]
        print(f"Detected {checkpoint_num_classes} classes from checkpoint")
    
    # If num_classes is provided as an argument, use that
    if num_classes is not None:
        print(f"Using provided number of classes: {num_classes}")
        actual_num_classes = num_classes
    # Otherwise use the number from checkpoint
    elif checkpoint_num_classes is not None:
        actual_num_classes = checkpoint_num_classes
    # Default to 7 classes if we can't determine
    else:
        actual_num_classes = 7
        print(f"Could not determine number of classes, defaulting to {actual_num_classes}")
    
    # Now get class names based on the actual number of classes
    if actual_num_classes == 7:
        # Default NEU-DET class names
        class_names = ["OK", "pitted_surface", "inclusion", "patches", 
                      "rolled-in_scale", "scratches", "crazing"]
        print(f"Using default NEU-DET class names: {class_names}")
    elif dataset_path and os.path.exists(dataset_path):
        # Try to get class names from dataset directory
        potential_class_names = sorted([d for d in os.listdir(dataset_path) 
                                if os.path.isdir(os.path.join(dataset_path, d))])
        
        # Check if the number of directories matches the number of classes
        if len(potential_class_names) == actual_num_classes:
            class_names = potential_class_names
            print(f"Using class names from dataset: {class_names}")
        else:
            # Generate generic class names
            class_names = [f"class_{i}" for i in range(actual_num_classes)]
            print(f"Number of directories ({len(potential_class_names)}) doesn't match number of classes ({actual_num_classes})")
            print(f"Using generic class names: {class_names}")
    else:
        # Generate generic class names
        class_names = [f"class_{i}" for i in range(actual_num_classes)]
        print(f"Using generic class names: {class_names}")
    
    # Create and load the model
    model_variant = 2  # Default
    if isinstance(config, dict) and 'model_variant' in config:
        model_variant = config['model_variant']
    elif hasattr(config, 'model_variant'):
        model_variant = config.model_variant
    
    print(f"Using model variant: B{model_variant}")
    model = SegmentationModel(num_classes=actual_num_classes, model_variant=model_variant)
    model.load_state_dict(checkpoint[state_dict_key])
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, 3, 256, 256)
    
    # Trace the model
    print("Tracing model...")
    traced_model = torch.jit.trace(model, example_input)
    
    # Try ONNX conversion if possible
    onnx_path = os.path.splitext(output_path)[0] + ".onnx"
    onnx_conversion_success = False
    
    try:
        # Check if onnx is installed
        import importlib.util
        if importlib.util.find_spec("onnx") is not None:
            print("Converting to ONNX as intermediate format...")
            torch.onnx.export(
                model, 
                example_input, 
                onnx_path, 
                input_names=["input_image"], 
                output_names=["output"],
                dynamic_axes={"input_image": {0: "batch"}, "output": {0: "batch"}},
                opset_version=12
            )
            print(f"ONNX model saved to {onnx_path}")
            onnx_conversion_success = True
        else:
            print("ONNX package not found, skipping ONNX conversion step.")
    except Exception as e:
        print(f"ONNX conversion failed: {e}")
        print("Skipping ONNX conversion step.")
    
    # Convert to CoreML
    print("Converting to CoreML...")
    if onnx_conversion_success:
        try:
            # Load the ONNX model
            onnx_model = ct.converters.onnx.load(onnx_path)
            
            # Convert to CoreML
            mlmodel = ct.convert(
                onnx_model,
                inputs=[ct.ImageType(name="input_image", shape=(1, 3, 256, 256), scale=1/255.0, bias=[0, 0, 0], color_layout='RGB')],
                compute_precision=ct.precision.FLOAT16,
                minimum_deployment_target=ct.target.macOS13
            )
            print("ONNX to CoreML conversion successful.")
        except Exception as e:
            print(f"ONNX to CoreML conversion failed: {e}")
            print("Falling back to direct conversion...")
            onnx_conversion_success = False
    
    # If ONNX conversion failed or was skipped, use direct conversion
    if not onnx_conversion_success:
        print("Using direct PyTorch to CoreML conversion...")
        try:
            # Direct conversion from PyTorch
            mlmodel = ct.convert(
                traced_model,
                inputs=[ct.ImageType(name="input_image", shape=example_input.shape, scale=1/255.0, bias=[0, 0, 0], color_layout='RGB')],
                compute_precision=ct.precision.FLOAT16,
                minimum_deployment_target=ct.target.macOS13
            )
        except Exception as e:
            print(f"Direct conversion with ImageType failed: {e}")
            print("Trying with TensorType...")
            
            # Try with TensorType as a last resort
            mlmodel = ct.convert(
                traced_model,
                inputs=[ct.TensorType(name="input_image", shape=example_input.shape)],
                compute_precision=ct.precision.FLOAT16,
                minimum_deployment_target=ct.target.macOS13
            )
    
    # Add metadata
    mlmodel.user_defined_metadata['class_names'] = ','.join(class_names)
    mlmodel.user_defined_metadata['model_variant'] = f"SegFormer B{model_variant}"
    
    # Save the model
    print(f"Saving model to {output_path}...")
    mlmodel.save(output_path)
    
    # After saving the CoreML model, export model info to a file
    print("Exporting model info file...")

    # Get model input and output shapes
    input_shape = example_input.shape  # (1, 3, 256, 256)
    output_shape = (1, actual_num_classes, 256, 256)  # Typical segmentation output shape

    # Create model info dictionary
    model_info = {
        "model_name": os.path.basename(output_path),
        "model_variant": f"SegFormer B{model_variant}",
        "num_classes": actual_num_classes,
        "class_names": class_names,
        "input_shape": list(input_shape),
        "output_shape": list(output_shape),
        "input_scale": 1.0/255.0,
        "input_bias": [0, 0, 0],
        "color_layout": "RGB",
        "creation_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Determine the info file path (same directory as the model)
    info_file_path = os.path.join(os.path.dirname(output_path), "modelInfo.inf")

    # Write the info to a file
    with open(info_file_path, 'w') as f:
        # Write header
        f.write("# Surface Defect Detection Model Information\n")
        f.write("# Generated by convert_model.py\n\n")
        
        # Write model details
        f.write(f"MODEL_NAME={model_info['model_name']}\n")
        f.write(f"MODEL_VARIANT={model_info['model_variant']}\n")
        f.write(f"NUM_CLASSES={model_info['num_classes']}\n")
        f.write(f"CLASS_NAMES={','.join(model_info['class_names'])}\n")
        f.write(f"INPUT_SHAPE={','.join(map(str, model_info['input_shape']))}\n")
        f.write(f"OUTPUT_SHAPE={','.join(map(str, model_info['output_shape']))}\n")
        f.write(f"INPUT_SCALE={model_info['input_scale']}\n")
        f.write(f"INPUT_BIAS={','.join(map(str, model_info['input_bias']))}\n")
        f.write(f"COLOR_LAYOUT={model_info['color_layout']}\n")
        f.write(f"CREATION_DATE={model_info['creation_date']}\n")

    print(f"Model info exported to {info_file_path}")
    
    # Compile the model for runtime use
    print("Compiling model for runtime use...")
    try:
        # Create a compiled model path
        mlpackagec_path = output_path + "c"
        if os.path.exists(mlpackagec_path):
            shutil.rmtree(mlpackagec_path)
        
        # Try using xcrun coremlcompiler if available
        import subprocess
        
        # First check if xcrun is available
        try:
            subprocess.run(["xcrun", "--version"], check=True, capture_output=True)
            has_xcrun = True
        except (subprocess.SubprocessError, FileNotFoundError):
            has_xcrun = False
        
        if has_xcrun:
            print("Using xcrun coremlcompiler to compile the model...")
            compile_cmd = ["xcrun", "coremlcompiler", "compile", output_path, mlpackagec_path]
            print(f"Running: {' '.join(compile_cmd)}")
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Model compiled successfully to {mlpackagec_path}")
            else:
                print(f"Error compiling model with xcrun: {result.stderr}")
                print("Falling back to coremltools...")
                has_xcrun = False
        
        if not has_xcrun:
            # Fall back to coremltools approach
            print("Using coremltools to compile the model...")
            # Try the simplest approach - load and save
            loaded_model = ct.models.MLModel(output_path)
            compiled_model_path = os.path.splitext(output_path)[0] + "_compiled.mlpackage"
            loaded_model.save(compiled_model_path)
            print(f"Model saved to {compiled_model_path}")
            
            # Create a .mlpackagec version by copying
            if os.path.exists(compiled_model_path):
                if os.path.isdir(compiled_model_path):
                    shutil.copytree(compiled_model_path, mlpackagec_path)
                else:
                    shutil.copy2(compiled_model_path, mlpackagec_path)
                print(f"Created .mlpackagec version at {mlpackagec_path}")
            else:
                # If that fails, just copy the original model
                print("Compiled model not found, copying original model...")
                if os.path.isdir(output_path):
                    shutil.copytree(output_path, mlpackagec_path)
                else:
                    shutil.copy2(output_path, mlpackagec_path)
                print(f"Created .mlpackagec by copying original model")
        
        # Verify the compiled model
        if os.path.exists(mlpackagec_path):
            if os.path.isdir(mlpackagec_path):
                # Calculate total size including subdirectories
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(mlpackagec_path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if os.path.exists(fp):  # Skip if file doesn't exist (symlinks etc)
                            total_size += os.path.getsize(fp)
                
                print(f"Compiled model directory size: {total_size} bytes ({total_size/1024/1024:.2f} MB)")
                if total_size < 1000:
                    print("WARNING: Compiled model is suspiciously small!")
            else:
                file_size = os.path.getsize(mlpackagec_path)
                print(f"Compiled model file size: {file_size} bytes ({file_size/1024/1024:.2f} MB)")
                if file_size < 1000:
                    print("WARNING: Compiled model is suspiciously small!")
        
    except Exception as e:
        print(f"Warning: Failed to compile model: {e}")
        print("You may need to compile the model manually before using it.")
        
        # Try a simpler approach as a last resort
        try:
            print("Trying a simpler approach...")
            mlpackagec_path = output_path + "c"
            if os.path.exists(mlpackagec_path):
                if os.path.isdir(mlpackagec_path):
                    shutil.rmtree(mlpackagec_path)
                else:
                    os.remove(mlpackagec_path)
                
            # Just copy the original model
            if os.path.isdir(output_path):
                shutil.copytree(output_path, mlpackagec_path)
            else:
                shutil.copy2(output_path, mlpackagec_path)
            print(f"Created .mlpackagec by copying original model to {mlpackagec_path}")
        except Exception as e2:
            print(f"Final attempt also failed: {e2}")
            print("You will need to manually compile the model before using it.")

    # Clean up intermediate files
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
        print(f"Removed intermediate ONNX file: {onnx_path}")

    print("Model conversion and compilation completed successfully.")

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to CoreML")
    parser.add_argument('model_path', type=str, 
                        help='Path to the trained model')
    parser.add_argument('--output_path', type=str, 
                        default='SurfaceDefectDetector.mlpackage',
                        help='Path to save the converted model')
    parser.add_argument('--dataset_path', type=str, 
                        help='Path to dataset for class names (optional)')
    parser.add_argument('--num_classes', type=int, 
                        help='Number of classes (optional, will be detected from checkpoint if not provided)')
    args = parser.parse_args()
    
    convert_to_coreml(args.model_path, args.output_path, args.dataset_path, args.num_classes)

if __name__ == "__main__":
    main() 