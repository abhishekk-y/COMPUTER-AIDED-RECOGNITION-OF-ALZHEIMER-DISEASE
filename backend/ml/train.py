"""
CARE-AD+ Model Training Script

Complete training pipeline for Alzheimer's disease classification model.
Includes:
- Training loop with progress tracking
- Validation and metrics logging
- Model checkpointing
- Early stopping
- Learning rate scheduling
"""
import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from datetime import datetime
from typing import Dict, Optional, Tuple
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.model import create_model, SimpleImageClassifier
from ml.dataset import create_data_loaders, get_class_weights, AlzheimerDataset


class Trainer:
    """
    Complete training pipeline for Alzheimer's classifier.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        class_weights: Optional[torch.Tensor] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function with class weights for imbalanced data
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Progress update
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Acc: {100.*correct/total:.2f}%")
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def save_checkpoint(self, path: str, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.replace('.pth', '_best.pth')
            torch.save(self.model.state_dict(), best_path)
            print(f"💾 Saved best model to {best_path}")
    
    def train(
        self,
        epochs: int = 50,
        save_dir: str = "../models",
        early_stopping_patience: int = 15
    ) -> Dict:
        """
        Complete training loop.
        
        Args:
            epochs: Number of training epochs
            save_dir: Directory to save models
            early_stopping_patience: Epochs to wait before early stopping
        
        Returns:
            Training history dictionary
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Training status file for API monitoring
        status_file = os.path.join(save_dir, "training_status.json")
        
        print(f"\n{'='*60}")
        print(f"🧠 CARE-AD+ Training Started")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            print(f"\n📅 Epoch {epoch}/{epochs}")
            print("-" * 40)
            
            # Update status file
            status = {
                "status": "training",
                "current_epoch": epoch,
                "total_epochs": epochs,
                "best_val_acc": self.best_val_acc,
                "message": f"Training epoch {epoch}/{epochs}"
            }
            with open(status_file, 'w') as f:
                json.dump(status, f)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\n📊 Epoch {epoch} Summary:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"   Time: {epoch_time:.1f}s | LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Check for best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.patience_counter = 0
                print(f"   🌟 New best validation accuracy!")
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            checkpoint_path = os.path.join(save_dir, f"alzheimer_cnn_epoch_{epoch}.pth")
            self.save_checkpoint(checkpoint_path, epoch, is_best)
            
            # Early stopping check
            if self.patience_counter >= early_stopping_patience:
                print(f"\n⏹️ Early stopping triggered after {early_stopping_patience} epochs without improvement")
                break
        
        # Training complete
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"✅ Training Complete!")
        print(f"{'='*60}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final model
        final_path = os.path.join(save_dir, "alzheimer_cnn.pth")
        torch.save(self.model.state_dict(), final_path)
        print(f"💾 Final model saved to {final_path}")
        
        # Save training history
        history_path = os.path.join(save_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Update status file
        status = {
            "status": "completed",
            "current_epoch": epoch,
            "total_epochs": epochs,
            "best_val_acc": self.best_val_acc,
            "best_val_loss": self.best_val_loss,
            "total_time_minutes": total_time / 60,
            "message": "Training completed successfully"
        }
        with open(status_file, 'w') as f:
            json.dump(status, f)
        
        return self.history


def train_model(
    dataset_dir: str = "../archive/combined_images",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    save_dir: str = "../models",
    image_size: int = 224
):
    """
    Main training function - can be called from API.
    
    Args:
        dataset_dir: Path to dataset
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        save_dir: Where to save models
        image_size: Image size
    """
    print(f"🚀 Starting CARE-AD+ model training...")
    print(f"   Dataset: {dataset_dir}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Create data loaders
    train_loader, val_loader, test_loader, dataset_info = create_data_loaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    # Calculate class weights
    class_weights = get_class_weights(dataset_info['class_distribution'])
    print(f"   Class weights: {class_weights.tolist()}")
    
    # Create model
    model = create_model(
        model_type="simple",
        num_classes=dataset_info['num_classes'],
        pretrained=True
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        class_weights=class_weights
    )
    
    # Train
    history = trainer.train(
        epochs=epochs,
        save_dir=save_dir,
        early_stopping_patience=15
    )
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    from ml.evaluate import evaluate_model
    
    model_path = os.path.join(save_dir, "alzheimer_cnn_best.pth")
    test_metrics = evaluate_model(
        model_path=model_path,
        test_loader=test_loader,
        class_names=dataset_info['class_names'],
        device=device,
        save_dir=save_dir
    )
    
    # Save complete metrics
    metrics = {
        "model_version": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "accuracy": test_metrics['accuracy'],
        "precision": test_metrics['precision'],
        "recall": test_metrics['recall'],
        "f1_score": test_metrics['f1_score'],
        "training_samples": dataset_info['train_samples'],
        "validation_samples": dataset_info['val_samples'],
        "test_samples": dataset_info['test_samples'],
        "class_distribution": dataset_info['class_distribution'],
        "confusion_matrix": test_metrics['confusion_matrix'].tolist()
    }
    
    metrics_path = os.path.join(save_dir, "model_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Training complete! Metrics saved to {metrics_path}")
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CARE-AD+ model")
    parser.add_argument("--dataset", default="../archive/combined_images",
                        help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--save-dir", default="../models",
                        help="Directory to save models")
    
    args = parser.parse_args()
    
    train_model(
        dataset_dir=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_dir=args.save_dir
    )
