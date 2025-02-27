import logging
from pathlib import Path
from anomalib.data import MVTec
from anomalib.models import Patchcore
from anomalib.engine import Engine
import torch
from lightning.pytorch import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PatchCore Training")

class PatchCoreTrainer:
    def __init__(self, save_path="./models", device=None):
        self.save_path = Path(save_path)
        self.save_path.mkdir(exist_ok=True)
        
        # Set device
        if device is None:
            self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")

    def train(self):
        """Train the PatchCore model"""
        logger.info("Starting training...")
        
        try:
            # Initialize MVTec dataset
            datamodule = MVTec(
                category="bottle",  # or "carpet", "leather", etc.
                image_size=(224, 224),
                train_batch_size=32,
                eval_batch_size=32,
                num_workers=4
            )
            
            # Initialize model
            model = Patchcore(
                backbone="wide_resnet50_2",
                layers=["layer2", "layer3"],
                coreset_sampling_ratio=0.1,
                pre_trained=True
            )
            
            # Initialize engine with minimal parameters
            engine = Engine()
            
            # Configure trainer parameters
            engine.trainer_args = {
                "accelerator": 'mps' if torch.backends.mps.is_available() else 'cpu',
                "devices": 1,
                "max_epochs": 1,
                "default_root_dir": str(self.save_path)
            }
            
            # Train model
            logger.info("Training model...")
            engine.fit(
                model=model,
                datamodule=datamodule
            )
            
            # Get predictions on test set
            logger.info("Running predictions on test set...")
            predictions = engine.predict(
                model=model,
                datamodule=datamodule
            )
            
            logger.info("Training and evaluation completed successfully")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

def main():
    # Training configuration
    save_path = "./models"      # Where to save the trained model
    
    # Initialize trainer
    trainer = PatchCoreTrainer(
        save_path=save_path
    )
    
    # Train the model
    trainer.train()

if __name__ == "__main__":
    main() 