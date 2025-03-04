#!/usr/bin/env python3
import os
import sys
import shutil
import subprocess

def compile_model(model_path):
    """Compile a CoreML model using xcrun."""
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return False
    
    # Create output path
    output_path = model_path + "c"
    if os.path.exists(output_path):
        if os.path.isdir(output_path):
            shutil.rmtree(output_path)
        else:
            os.remove(output_path)
    
    # Try using xcrun
    try:
        print(f"Compiling model {model_path} to {output_path}...")
        cmd = ["xcrun", "coremlcompiler", "compile", model_path, output_path]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Model compiled successfully to {output_path}")
            
            # Check the size of the compiled model
            if os.path.isdir(output_path):
                size = get_dir_size(output_path)
                print(f"Compiled model directory size: {size} bytes ({size/1024/1024:.2f} MB)")
                if size < 1000:
                    print("WARNING: Compiled model is suspiciously small!")
            else:
                size = os.path.getsize(output_path)
                print(f"Compiled model file size: {size} bytes ({size/1024/1024:.2f} MB)")
            
            return True
        else:
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"Compilation failed: {e}")
        
        # Fall back to simple copy
        print("Falling back to simple copy...")
        try:
            if os.path.isdir(model_path):
                shutil.copytree(model_path, output_path)
            else:
                shutil.copy2(model_path, output_path)
            print(f"Copied model to {output_path}")
            return True
        except Exception as e2:
            print(f"Copy also failed: {e2}")
            return False

# Add this function to calculate directory size
def get_dir_size(path):
    """Calculate the total size of a directory including all subdirectories."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):  # Skip if file doesn't exist (symlinks etc)
                total_size += os.path.getsize(fp)
    return total_size

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compile_model.py <path_to_model>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    success = compile_model(model_path)
    sys.exit(0 if success else 1) 