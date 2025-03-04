import coremltools as ct
import sys
import os

def convert_to_mlmodel(package_path):
    try:
        # Load the mlpackage
        model = ct.models.MLModel(package_path)
        
        # Save as mlmodel
        mlmodel_path = os.path.splitext(package_path)[0] + ".mlmodel"
        model.save(mlmodel_path)
        
        print(f"Converted to {mlmodel_path}")
        return mlmodel_path
    except Exception as e:
        print(f"Error converting model: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_to_mlmodel.py <path_to_mlpackage>")
        sys.exit(1)
    
    package_path = sys.argv[1]
    convert_to_mlmodel(package_path) 