"""
CARE-AD+ Model Evaluation

Comprehensive evaluation metrics for Alzheimer's classification model.
Includes:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC Curves
- Per-class metrics
- Visualization generation
"""
import os
import torch
import torch.nn as nn
import numpy as np
import json
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from ml.model import load_model, SimpleImageClassifier


def evaluate_model(
    model_path: str,
    test_loader,
    class_names: List[str],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "../models"
) -> Dict:
    """
    Comprehensive model evaluation.
    
    Args:
        model_path: Path to trained model weights
        test_loader: Test data loader
        class_names: List of class names
        device: Device to use
        save_dir: Directory to save evaluation results
    
    Returns:
        Dictionary with all metrics
    """
    print("\n📊 Model Evaluation")
    print("=" * 50)
    
    # Load model
    model = load_model(
        model_path=model_path,
        model_type="simple",
        num_classes=len(class_names),
        device=device
    )
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Collect predictions
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Print results
    print(f"\n📈 Overall Metrics:")
    print(f"   Accuracy:  {accuracy*100:.2f}%")
    print(f"   Precision: {precision*100:.2f}%")
    print(f"   Recall:    {recall*100:.2f}%")
    print(f"   F1-Score:  {f1*100:.2f}%")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    # Confusion matrix
    print(f"\n🔢 Confusion Matrix:")
    print(conf_matrix)
    
    # Calculate per-class metrics
    per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'precision': float(per_class_precision[i]),
            'recall': float(per_class_recall[i]),
            'f1_score': float(per_class_f1[i])
        }
    
    # Try to calculate AUC-ROC (for multi-class)
    try:
        auc_roc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except:
        auc_roc = None
    
    # Generate visualizations
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix - CARE-AD+')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    conf_matrix_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(conf_matrix_path, dpi=300)
    plt.close()
    print(f"💾 Confusion matrix saved to {conf_matrix_path}")
    
    # Plot per-class metrics
    plt.figure(figsize=(12, 6))
    x = np.arange(len(class_names))
    width = 0.25
    
    plt.bar(x - width, per_class_precision, width, label='Precision', color='#4F46E5')
    plt.bar(x, per_class_recall, width, label='Recall', color='#06B6D4')
    plt.bar(x + width, per_class_f1, width, label='F1-Score', color='#10B981')
    
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Per-Class Performance Metrics')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    metrics_path = os.path.join(save_dir, 'per_class_metrics.png')
    plt.savefig(metrics_path, dpi=300)
    plt.close()
    print(f"💾 Per-class metrics saved to {metrics_path}")
    
    # Create results dictionary
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_roc': float(auc_roc) if auc_roc is not None else None,
        'confusion_matrix': conf_matrix,
        'per_class_metrics': per_class_metrics,
        'total_samples': len(all_labels),
        'predictions': {
            'correct': int(np.sum(all_preds == all_labels)),
            'incorrect': int(np.sum(all_preds != all_labels))
        }
    }
    
    print(f"\n✅ Evaluation complete!")
    
    return results


def plot_training_history(history_path: str, save_dir: str = "../models"):
    """
    Plot training history curves.
    
    Args:
        history_path: Path to training_history.json
        save_dir: Directory to save plots
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 0].plot(epochs, history['learning_rates'], 'g-')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Best metrics annotation
    best_val_acc = max(history['val_acc'])
    best_epoch = history['val_acc'].index(best_val_acc) + 1
    
    axes[1, 1].axis('off')
    info_text = f"""
    Training Summary
    ================
    
    Total Epochs: {len(epochs)}
    Best Validation Accuracy: {best_val_acc:.2f}%
    Best Epoch: {best_epoch}
    
    Final Training Loss: {history['train_loss'][-1]:.4f}
    Final Validation Loss: {history['val_loss'][-1]:.4f}
    
    Final Training Accuracy: {history['train_acc'][-1]:.2f}%
    Final Validation Accuracy: {history['val_acc'][-1]:.2f}%
    """
    axes[1, 1].text(0.1, 0.5, info_text, fontsize=12, fontfamily='monospace',
                    verticalalignment='center', transform=axes[1, 1].transAxes)
    
    plt.suptitle('CARE-AD+ Training History', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    history_plot_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(history_plot_path, dpi=300)
    plt.close()
    
    print(f"💾 Training history plot saved to {history_plot_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate CARE-AD+ model")
    parser.add_argument("--model", required=True, help="Path to model weights")
    parser.add_argument("--dataset", default="../archive/combined_images",
                        help="Path to dataset")
    parser.add_argument("--save-dir", default="../models",
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    from ml.dataset import create_data_loaders, AlzheimerDataset
    
    # Create test loader
    _, _, test_loader, info = create_data_loaders(
        args.dataset,
        batch_size=32,
        num_workers=0
    )
    
    # Evaluate
    results = evaluate_model(
        model_path=args.model,
        test_loader=test_loader,
        class_names=info['class_names'],
        save_dir=args.save_dir
    )
    
    # Plot history if available
    history_path = os.path.join(args.save_dir, 'training_history.json')
    if os.path.exists(history_path):
        plot_training_history(history_path, args.save_dir)
