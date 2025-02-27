import torch
import numpy as np
import cv2
from anomalib.models import Patchcore
from anomalib.data.utils import read_image
from pathlib import Path
import os
import platform
from torchvision import transforms
from anomalib.engine import Engine

class SurfaceInspector:
    def __init__(self, model_path="./models/model.ckpt"):
        # Special handling for M1 chip
        if platform.processor() == 'arm':
            self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        print(f"Using device: {self.device}")
        
        # Initialize transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),          # Converts to [C, H, W] and scales to [0,1]
            transforms.Resize((224, 224), antialias=True),  # Fixed size 224x224
            transforms.Grayscale(num_output_channels=3),  # Convert to grayscale but keep 3 channels
            transforms.Normalize(
                mean=[0.485, 0.485, 0.485],  # Same value for all channels since it's grayscale
                std=[0.229, 0.229, 0.229]
            )
        ])
        
        # Initialize model
        self.model = Patchcore(
            backbone="wide_resnet50_2",
            pre_trained=True
        ).to(self.device)
        
        # Print model information
        print("\nModel Information:")
        print(f"Model type: {type(self.model)}")
        
        # Try to get input shape from model
        try:
            if hasattr(self.model, 'input_size'):
                print(f"Expected input size: {self.model.input_size}")
            if hasattr(self.model, 'backbone'):
                print(f"Backbone: {self.model.backbone}")
                # Print first layer's expected input
                first_layer = next(self.model.backbone.parameters())
                print(f"First layer shape: {first_layer.shape}")
        except Exception as e:
            print(f"Could not get model details: {e}")
        
        # Convert model parameters to float32
        self.model = self.model.float()
        self.model.eval()
        
        # Load trained model if available
        if model_path and os.path.exists(model_path):
            self.model = Engine.load_from_checkpoint(model_path).to(self.device)
            print(f"Loaded trained model from {model_path}")

    def inspect_image(self, image_path, threshold=0.5):
        try:
            # Load and process image
            image = read_image(image_path)
            print(f"\nShape information:")
            print(f"1. Original image shape: {image.shape}")  # Should be [H, W, 3]
            
            # Print intermediate shapes during transformation
            temp_tensor = transforms.ToTensor()(image)
            print(f"2. After ToTensor: {temp_tensor.shape}")
            
            temp_tensor = transforms.Resize((224, 224), antialias=True)(temp_tensor)
            print(f"3. After Resize: {temp_tensor.shape}")
            
            # Final input tensor
            image_tensor = self.transform(image).float()
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension [1, 3, 224, 224]
            image_tensor = image_tensor.to(self.device)
            print(f"5. Final input tensor shape: {image_tensor.shape}")
            print(f"   Tensor device: {image_tensor.device}")
            print(f"   Model device: {next(self.model.parameters()).device}")
            
            # Get model's forward function signature
            # print("\nModel forward function info:")
            # print(f"Forward function: {self.model.forward.__doc__}")
            
            # Get predictions
            with torch.no_grad():
                predictions = self.model(image_tensor)
            
            # Extract results and ensure float32
            anomaly_map = predictions.anomaly_map.float()
            print(f"Anomaly map shape before processing: {anomaly_map.shape}")
            
            # Handle different possible shapes of anomaly map
            if len(anomaly_map.shape) == 4:  # [B, C, H, W]
                anomaly_map = anomaly_map[0]  # Remove batch dimension
            if len(anomaly_map.shape) == 3:  # [C, H, W]
                anomaly_map = anomaly_map[0]  # Take first channel
            
            # Convert to numpy and ensure correct type
            anomaly_map = anomaly_map.cpu().numpy()
            print(f"Anomaly map shape after processing: {anomaly_map.shape}")
            
            pred_score = predictions.pred_score.float().cpu().item()
            
            # Ensure anomaly map is float32
            anomaly_map = anomaly_map.astype(np.float32)
            
            # Resize anomaly map to match original image size
            anomaly_map = cv2.resize(
                anomaly_map,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            
            # Create heat map
            normalized_map = ((anomaly_map - anomaly_map.min()) / 
                            (anomaly_map.max() - anomaly_map.min() + 1e-8))
            heat_map = cv2.applyColorMap(
                (normalized_map * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            
            # Create binary map
            binary_map = (normalized_map > threshold).astype(np.uint8) * 255
            
            # Prepare visualization
            image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            result = image_cv.copy()
            
            # Find and draw contours
            contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, (0, 0, 255), 2)
            
            return result, heat_map, binary_map, pred_score
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            raise

def test_marble_dataset(inspector, dataset_path, num_samples=5):
    # Test on good samples
    good_path = os.path.join(dataset_path, "good")
    good_images = list(Path(good_path).glob("*.[jp][pn][g]"))
    if num_samples > 0:
        good_images = good_images[:num_samples]
    
    print(f"\nTesting normal (good) samples:")
    for img_path in good_images:
        result, heat_map, binary_map, score = inspector.inspect_image(str(img_path))
        print(f"{img_path.name}: Anomaly score = {score:.4f}")
        
        # Display results
        # cv2.imshow("Normal Sample", result)
        # cv2.imshow("Heat Map", (heat_map * 255).astype(np.uint8))
        # cv2.imshow("Binary Map", binary_map)
        
        # key = cv2.waitKey(0)
        # if key == ord('q'):
        #     break
    
    # Test on defective samples
    defect_path = os.path.join(dataset_path, "defect")
    defect_images = list(Path(defect_path).glob("*.[jp][pn][g]"))
    if num_samples > 0:
        defect_images = defect_images[:num_samples]
    
    print(f"\nTesting defective samples:")
    for img_path in defect_images:
        result, heat_map, binary_map, score = inspector.inspect_image(str(img_path))
        print(f"{img_path.name}: Anomaly score = {score:.4f}")
        
        # Display results
        cv2.imshow("Defective Sample", result)
        cv2.imshow("Heat Map", (heat_map * 255).astype(np.uint8))
        cv2.imshow("Binary Map", binary_map)
        
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

def main():
    try:
        inspector = SurfaceInspector()
    except Exception as e:
        print(f"Error initializing inspector: {str(e)}")
        return

    marble_path = "./marble"
    
    if not os.path.exists(marble_path):
        print(f"Error: Dataset path not found: {marble_path}")
        return
        
    print("\nStarting marble surface inspection...")
    print("Press any key to continue to next image")
    print("Press 'q' to quit\n")
    
    try:
        test_marble_dataset(inspector, marble_path)
    except Exception as e:
        print(f"Error during inspection: {str(e)}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 