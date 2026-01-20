"""
CARE-AD+ ML Service

Service layer for machine learning predictions.
Handles model loading, preprocessing, and inference.
"""
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Dict, Optional, Any
import sys

# Add ML directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ml'))

from app.config import settings


class MLService:
    """
    Machine Learning service for Alzheimer's disease prediction.
    """
    
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.class_names = settings.CLASS_NAMES
        self.image_size = settings.IMAGE_SIZE
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model."""
        try:
            from ml.model import load_model, create_model
            
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                settings.MODEL_PATH.lstrip('../')
            )
            
            if os.path.exists(model_path):
                self.model = load_model(
                    model_path=model_path,
                    model_type="simple",
                    num_classes=settings.NUM_CLASSES,
                    device=self.device
                )
                print(f"✅ Model loaded from {model_path}")
            else:
                # Create untrained model for demo purposes
                self.model = create_model(
                    model_type="simple",
                    num_classes=settings.NUM_CLASSES,
                    pretrained=True
                ).to(self.device)
                self.model.eval()
                print(f"⚠️ No trained model found. Using pretrained backbone for demo.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess an image for model input.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image)
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def predict(
        self,
        image_path: str,
        clinical_features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make a prediction for an MRI image.
        
        Args:
            image_path: Path to the MRI image
            clinical_features: Optional clinical features dict
        
        Returns:
            Prediction result with class, confidence, and probabilities
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Preprocess image
        image_tensor = self.preprocess_image(image_path).to(self.device)
        
        # Run inference
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = probabilities.max(1)
        
        # Get class name
        predicted_class = self.class_names[predicted_idx.item()]
        
        # Create probabilities dict
        prob_dict = {
            class_name: float(probabilities[0, i])
            for i, class_name in enumerate(self.class_names)
        }
        
        result = {
            "predicted_class": predicted_class,
            "predicted_index": int(predicted_idx.item()),
            "confidence": float(confidence.item()),
            "probabilities": prob_dict,
            "device": self.device
        }
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"status": "not_loaded"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        
        return {
            "status": "loaded",
            "device": self.device,
            "num_classes": settings.NUM_CLASSES,
            "class_names": self.class_names,
            "image_size": self.image_size,
            "total_parameters": total_params
        }
    
    def reload_model(self):
        """Reload the model (e.g., after retraining)."""
        self._load_model()
        return self.get_model_info()


# Singleton instance
_ml_service = None

def get_ml_service() -> MLService:
    """Get or create the ML service singleton."""
    global _ml_service
    if _ml_service is None:
        _ml_service = MLService()
    return _ml_service
