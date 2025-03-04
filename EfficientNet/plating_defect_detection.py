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
from datetime import datetime
from dataset import PlatingDefectDataset

class PlatingDefectDetector:
    def __init__(self, model_path=None):
        # Load class names from dataset instead of hardcoding
        self.defect_types = load_class_names()
        self.num_classes = len(self.defect_types)
        
        print(f"Loaded {self.num_classes} classes: {self.defect_types}")
        
        # Initialize device
        self.device = torch.device('mps' if torch.backends.mps.is_available() 
                                 else 'cuda' if torch.cuda.is_available() 
                                 else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = self._load_model(model_path)
        
        # Initialize transform without normalization
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def _load_model(self, model_path):
        """Load the model with proper configuration"""
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            model = SegmentationModel.load_model(model_path, device=self.device)
        else:
            print("Warning: No model path provided or model not found. Using untrained model.")
            model = SegmentationModel(num_classes=self.num_classes).to(self.device)
        
        model.eval()
        return model

    def detect_defect(self, image_path, threshold=0.5, ok_sensitivity=1.0):
        """
        Detect defects in an image with adjustable sensitivity
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Transform image (now in [0,1] range for SegFormer)
        image_tensor = self.transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(image_tensor)
            # Apply softmax to get probabilities
            probabilities = torch.softmax(output, dim=1)
            
            # Adjust OK class probability
            probabilities[:, 0] *= ok_sensitivity
            probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
            
            # Get predictions
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
        
        # Calculate grid layout based on number of classes
        num_prob_plots = len(self.defect_types)
        
        # Determine grid layout dynamically
        # We need at least 2 plots (original + overlay) plus class probability maps
        total_plots = 2 + num_prob_plots
        
        if total_plots <= 6:
            num_rows, num_cols = 2, 3
        elif total_plots <= 8:
            num_rows, num_cols = 2, 4
        elif total_plots <= 10:
            num_rows, num_cols = 2, 5
        else:
            num_rows, num_cols = 3, 5  # Can accommodate up to 15 plots
        
        plt.figure(figsize=(num_cols * 5, num_rows * 5))  # Adjust figure size based on grid
        
        # Plot original image
        plt.subplot(num_rows, num_cols, 1)
        plt.imshow(original)
        plt.title('Original Image')
        plt.axis('off')
        
        # Create color map for different defect types - dynamically based on number of classes
        colors = []
        # First class (OK) is transparent
        colors.append([0, 0, 0, 0])  
        
        # Basic colors for up to 6 defect classes
        base_colors = [
            [1, 0, 0, 0.7],    # Red
            [0, 1, 0, 0.7],    # Green
            [0, 0, 1, 0.7],    # Blue
            [1, 1, 0, 0.7],    # Yellow
            [1, 0, 1, 0.7],    # Magenta
            [0, 1, 1, 0.7],    # Cyan
        ]
        
        # Add colors for each class
        for i in range(1, self.num_classes):
            if i <= len(base_colors):
                colors.append(base_colors[i-1])
            else:
                # Generate random colors for any additional classes
                colors.append([random.random(), random.random(), random.random(), 0.7])
        
        # Create overlay mask with alpha channel
        h, w, _ = original_np.shape
        overlay = np.zeros((h, w, 4), dtype=np.float32)  # RGBA
        
        # Add colors for each class
        for class_idx in range(self.num_classes):
            mask = pred_mask == class_idx
            if mask.any() and class_idx > 0:  # Skip OK class (index 0)
                color = np.array(colors[class_idx])
                for i in range(4):  # Apply RGBA channels
                    overlay[..., i][mask] = color[i]
        
        # Plot segmentation result
        plt.subplot(num_rows, num_cols, 2)
        plt.imshow(original)  # Plot original image first
        plt.imshow(overlay, alpha=overlay[..., 3])  # Overlay with transparency
        plt.title('Segmentation Overlay')
        plt.axis('off')
        
        # Add colorbar legend for defect types (skip OK class)
        legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=colors[i][:3], alpha=0.7) 
                          for i in range(1, self.num_classes)]  # Skip OK class
        plt.legend(legend_elements, self.defect_types[1:],  # Skip OK in legend
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
            plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)
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
    test_folder = 'NEU-DET/validation/images'
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

    # Save the final model with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'model_{timestamp}.pth'
    detector.model.save_model(save_path)  # Use new save method
    print(f"Model saved to {save_path}")

def load_class_names():
    """Load class names from dataset"""
    # Initialize minimal dataset object just to get class mapping
    dataset = PlatingDefectDataset(root_dir='NEU-DET', transform=None, train=False)
    class_names = dataset.get_class_names()
    return class_names

def predict_defect(image_path, model_path='best_model.pth'):
    """Predict defects on a single image"""
    # Load class names
    class_names = load_class_names()
    print(f"Using classes: {class_names}")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SegmentationModel(num_classes=len(class_names))
    model = SegmentationModel.load_model(model_path, device)
    model.eval()
    
    # ... rest of your prediction code ...
    
    # Use class_names for visualization and reporting
    # For example:
    pred_class_idx = prediction.argmax().item()
    defect_name = class_names[pred_class_idx]
    print(f"Predicted defect: {defect_name}")

def classify_image(image_path, model_path='best_model.pth', sensitivity=1.0):
    """
    Complete image classification function that returns class probabilities and visualizes results
    
    Args:
        image_path: Path to the image to classify
        model_path: Path to the trained model
        sensitivity: OK class sensitivity (lower = more sensitive to defects)
    
    Returns:
        dict: Classification results with class probabilities and path to visualization
    """
    detector = PlatingDefectDetector(model_path)
    
    # Detect defects
    pred_mask, prob_masks = detector.detect_defect(
        image_path, 
        ok_sensitivity=sensitivity
    )
    
    # Get dominant class
    unique_classes, counts = np.unique(pred_mask, return_counts=True)
    dominant_class_idx = unique_classes[np.argmax(counts)]
    dominant_class = detector.defect_types[dominant_class_idx]
    
    # Calculate confidence scores
    class_areas = {}
    for class_idx, class_name in enumerate(detector.defect_types):
        pixel_count = np.sum(pred_mask == class_idx)
        percentage = (pixel_count / pred_mask.size) * 100
        class_areas[class_name] = percentage
    
    # Create results directory
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Create visualization
    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(results_dir, f"{basename}_result.png")
    detector.visualize_results(image_path, pred_mask, prob_masks, save_path=output_path)
    
    # Return results dictionary
    return {
        'dominant_class': dominant_class,
        'class_percentages': class_areas,
        'visualization_path': output_path
    }

if __name__ == '__main__':
    main() 