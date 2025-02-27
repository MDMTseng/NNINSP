import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import timm
import matplotlib.pyplot as plt
import glob
import time
import random
import argparse
from model import SegmentationModel

class PlatingDefectDetector:
    def __init__(self, model_path=None):
        # Define defect types with OK as first class
        self.defect_types = ['OK', 'crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
        self.num_classes = len(self.defect_types)
        
        # Initialize device
        self.device = torch.device('mps' if torch.backends.mps.is_available() 
                                 else 'cuda' if torch.cuda.is_available() 
                                 else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = self._load_model(model_path)
        
        # Initialize transform
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self, model_path):
        model = SegmentationModel(num_classes=self.num_classes)
        
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print("Warning: No model path provided or model not found. Using untrained model.")
        
        model = model.to(self.device)
        model.eval()
        return model

    def detect_defect(self, image_path, threshold=0.5, ok_sensitivity=1.0):
        """
        Detect defects in an image with adjustable sensitivity
        Args:
            image_path: Path to the image
            threshold: Probability threshold for defect detection
            ok_sensitivity: Value to multiply OK class probabilities (lower values make detection more sensitive)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Transform image
        image_tensor = self.transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(image_tensor)
            # Apply softmax to get probabilities
            probabilities = torch.softmax(output, dim=1)
            
            # Adjust OK class (index 0) probability by sensitivity factor
            probabilities[:, 0] *= ok_sensitivity
            
            # Renormalize probabilities after adjustment
            probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
            
            # Get predicted class for each pixel (highest probability)
            predictions = torch.argmax(probabilities, dim=1)
            
            # Convert predictions to numpy
            pred_mask = predictions[0].cpu().numpy()
            prob_masks = probabilities[0].cpu().numpy()
            
            # Resize masks back to original size
            resized_pred = Image.fromarray(pred_mask.astype(np.uint8))
            resized_pred = resized_pred.resize(original_size, Image.Resampling.NEAREST)
            pred_mask = np.array(resized_pred)
            
            # Resize probability masks
            resized_probs = []
            for prob_mask in prob_masks:
                prob_img = Image.fromarray((prob_mask * 255).astype(np.uint8))
                prob_img = prob_img.resize(original_size, Image.Resampling.BILINEAR)
                resized_probs.append(np.array(prob_img) / 255.0)
            
            prob_masks = np.stack(resized_probs)
            
            return pred_mask, prob_masks

    def visualize_results(self, image_path, pred_mask, prob_masks, save_path=None):
        # Load original image
        original = Image.open(image_path).convert('RGB')
        original_np = np.array(original)
        
        # Create figure with subplots - Updated for 7 classes (OK + 6 defects)
        num_rows = 2
        num_cols = 5  # Increased from 4 to 5 to fit all probability maps
        plt.figure(figsize=(25, 10))  # Increased width to accommodate more subplots
        
        # Plot original image
        plt.subplot(num_rows, num_cols, 1)
        plt.imshow(original)
        plt.title('Original Image')
        plt.axis('off')
        
        # Create color map for different defect types
        colors = [
            [0.5, 0.5, 0.5],  # Gray for OK
            [1, 0, 0],        # Red for crazing
            [0, 1, 0],        # Green for inclusion
            [0, 0, 1],        # Blue for patches
            [1, 1, 0],        # Yellow for pitted_surface
            [1, 0, 1],        # Magenta for rolled-in_scale
            [0, 1, 1],        # Cyan for scratches
        ]
        
        # Create overlay mask
        h, w, c = original_np.shape
        overlay = np.zeros((h, w, c), dtype=np.float32)
        
        # Add colors for each class
        for class_idx in range(self.num_classes):
            mask = pred_mask == class_idx
            if mask.any():  # Only add color if this class is present
                color = np.array(colors[class_idx])
                for i in range(3):  # Apply color to RGB channels
                    overlay[..., i][mask] = color[i]
        
        # Blend original image with overlay
        alpha = 0.5  # Transparency of the overlay
        blended = (original_np * (1 - alpha) + overlay * 255 * alpha).astype(np.uint8)
        
        # Plot blended image
        plt.subplot(num_rows, num_cols, 2)
        plt.imshow(blended)
        plt.title('Segmentation Overlay')
        plt.axis('off')
        
        # Add colorbar legend for defect types
        legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=colors[i]) 
                          for i in range(self.num_classes)]
        plt.legend(legend_elements, self.defect_types, 
                  loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Plot probability maps for each defect type
        for i, (defect_type, prob_mask) in enumerate(zip(self.defect_types, prob_masks)):
            plt.subplot(num_rows, num_cols, i + 3)
            plt.imshow(prob_mask, cmap='jet', vmin=0, vmax=1)
            plt.title(f'{defect_type}\nProbability Map')
            plt.axis('off')
            plt.colorbar()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Results saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

    def test_inference_speed(self, image_path, num_iterations=100):
        """
        Test inference speed on a single image
        """
        # Load and preprocess image once
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # Warm up
        print("Warming up...")
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(image_tensor)
        
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize()
        
        # Measure inference time
        print(f"\nMeasuring inference speed over {num_iterations} iterations...")
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(image_tensor)
                
                if self.device.type == "mps":
                    torch.mps.synchronize()
                elif self.device.type == "cuda":
                    torch.cuda.synchronize()
        
        end_time = time.time()
        
        # Calculate statistics
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        fps = 1.0 / avg_time
        
        print(f"\nInference Speed Statistics:")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per image: {avg_time*1000:.2f} ms")
        print(f"FPS: {fps:.2f}")
        
        return fps

def process_test_folder(model_path, test_folder, output_folder, num_samples=10, threshold=0.5, ok_sensitivity=1.0):
    """
    Process randomly selected images from test folder
    
    Args:
        model_path: Path to the trained model
        test_folder: Folder containing test images
        output_folder: Folder to save results
        num_samples: Number of random images to process (default: 10)
        threshold: Probability threshold for confident predictions
        ok_sensitivity: Sensitivity factor for OK class (lower values = more sensitive detection)
    """
    # Initialize detector
    detector = PlatingDefectDetector(model_path=model_path)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.bmp', '*.png']:
        image_files.extend(glob.glob(os.path.join(test_folder, '**', ext), recursive=True))
    
    total_images = len(image_files)
    print(f"Found {total_images} total images")
    
    # Randomly sample images
    if num_samples > total_images:
        print(f"Warning: Requested {num_samples} samples but only {total_images} images available")
        num_samples = total_images
    
    selected_images = random.sample(image_files, num_samples)
    
    # Process each selected image
    for i, image_path in enumerate(selected_images, 1):
        print(f"\nProcessing image {i}/{num_samples}: {image_path}")
        
        # Get predictions with threshold and sensitivity
        pred_mask, prob_masks = detector.detect_defect(
            image_path, 
            threshold=threshold,
            ok_sensitivity=ok_sensitivity
        )
        
        # Create output filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_folder, f"{base_name}_result.png")
        
        # Get defect type from path
        defect_type = os.path.basename(os.path.dirname(image_path))
        print(f"True defect type: {defect_type}")
        
        # Calculate prediction statistics
        pred_classes = np.unique(pred_mask)
        class_percentages = {}
        # for class_idx in pred_classes:
        #     if class_idx == -1:
        #         percentage = (pred_mask == -1).mean() * 100
        #         print(f"No confident prediction: {percentage:.1f}%")
        #     else:
        #         percentage = (pred_mask == class_idx).mean() * 100
        #         defect_name = detector.defect_types[class_idx]
        #         class_percentages[defect_name] = percentage
        #         print(f"Predicted {defect_name}: {percentage:.1f}%")
        
        # Visualize and save results
        detector.visualize_results(image_path, pred_mask, prob_masks, save_path=output_path)
        print(f"Results saved to {output_path}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Plating Defect Detection Testing')
    parser.add_argument('-fps', action='store_true', help='Run speed test only')
    parser.add_argument('-n', type=int, default=10, help='Number of random images to test (default: 10)')
    parser.add_argument('-f', '--file', type=str, help='Path to single image for inference')
    parser.add_argument('-s', '--sensitivity', type=float, default=1.0, 
                        help='Detection sensitivity (lower values = more sensitive, default: 1.0)')
    args = parser.parse_args()
    
    # Set paths
    model_path = 'best_model.pth'
    test_folder = 'NEU-DET/train/images'
    output_folder = 'results'
    
    # Initialize detector
    detector = PlatingDefectDetector(model_path=model_path)
    
    if args.file:
        # Process single image
        print(f"\nProcessing single image: {args.file}")
        if not os.path.exists(args.file):
            print(f"Error: File {args.file} does not exist!")
            return
            
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Get predictions with sensitivity
        pred_mask, prob_masks = detector.detect_defect(
            args.file, 
            threshold=0.5,
            ok_sensitivity=args.sensitivity
        )
        
        # Create output filename
        base_name = os.path.splitext(os.path.basename(args.file))[0]
        output_path = os.path.join(output_folder, f"{base_name}_result.png")
        
        # Visualize and save results
        detector.visualize_results(args.file, pred_mask, prob_masks, save_path=output_path)
        print(f"Results saved to {output_path}")
        return
    
    # Get all image files for other modes
    image_files = []
    for ext in ['*.jpg', '*.bmp', '*.png']:
        image_files.extend(glob.glob(os.path.join(test_folder, '**', ext), recursive=True))
    
    if not image_files:
        print("No images found in test folder!")
        return
    
    # Select random test image
    test_image = random.choice(image_files)
    
    if args.fps:
        # Run speed test only
        print("\nRunning speed test...")
        print(f"Using {test_image} for speed test")
        fps = detector.test_inference_speed(test_image)
    else:
        print(f"\nProcessing {args.n} random images...")
        process_test_folder(
            model_path, 
            test_folder, 
            output_folder, 
            num_samples=args.n,
            ok_sensitivity=args.sensitivity
        )

if __name__ == '__main__':
    main() 