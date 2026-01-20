"""
CARE-AD+ CNN Model Architecture for Alzheimer's Disease Classification

This module implements a multi-modal deep learning model that combines:
1. A CNN for MRI image analysis (based on EfficientNet/ResNet)
2. A fully connected network for clinical features
3. A fusion layer for combined prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Dict, Tuple
import os


class ImageEncoder(nn.Module):
    """
    CNN encoder for MRI brain scan images.
    Uses transfer learning with EfficientNet-B0 backbone.
    """
    
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super(ImageEncoder, self).__init__()
        
        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        
        # Get the number of features from the classifier
        num_features = self.backbone.classifier[1].in_features
        
        # Remove the original classifier
        self.backbone.classifier = nn.Identity()
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Feature dimension
        self.feature_dim = num_features
        
        # Additional layers for feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from MRI image."""
        features = self.backbone(x)
        processed = self.feature_processor(features)
        return processed


class ClinicalEncoder(nn.Module):
    """
    Fully connected encoder for clinical features.
    
    Clinical features include:
    - Age, Gender, Education
    - MMSE Score (Mini-Mental State Examination)
    - CDR Score (Clinical Dementia Rating)
    - eTIV (Estimated Total Intracranial Volume)
    - nWBV (Normalized Whole Brain Volume)
    - ASF (Atlas Scaling Factor)
    """
    
    def __init__(self, num_features: int = 8):
        super(ClinicalEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        self.feature_dim = 32
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode clinical features."""
        return self.encoder(x)


class AlzheimerClassifier(nn.Module):
    """
    Complete multi-modal classifier for Alzheimer's disease detection.
    
    Combines image features and clinical features for classification.
    
    Classes:
    0 - MildDemented
    1 - ModerateDemented
    2 - NonDemented
    3 - VeryMildDemented
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        num_clinical_features: int = 8,
        use_clinical: bool = True,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super(AlzheimerClassifier, self).__init__()
        
        self.use_clinical = use_clinical
        self.num_classes = num_classes
        
        # Image encoder
        self.image_encoder = ImageEncoder(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
        
        # Clinical encoder (optional)
        if use_clinical:
            self.clinical_encoder = ClinicalEncoder(num_clinical_features)
            fusion_dim = self.image_encoder.feature_processor[-2].out_features + \
                        self.clinical_encoder.feature_dim
        else:
            self.clinical_encoder = None
            fusion_dim = self.image_encoder.feature_processor[-2].out_features
        
        # Fusion and classification layers
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Final classifier
        self.classifier = nn.Linear(64, num_classes)
        
        # For Grad-CAM - store the last conv layer
        self.gradients = None
        self.activations = None
    
    def activations_hook(self, grad):
        """Hook for gradients."""
        self.gradients = grad
    
    def get_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get activations from the last convolutional layer."""
        # Access the last conv layer of EfficientNet
        x = self.image_encoder.backbone.features(x)
        self.activations = x
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        return x
    
    def forward(
        self,
        image: torch.Tensor,
        clinical_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            image: MRI image tensor [B, 3, H, W]
            clinical_features: Clinical features tensor [B, num_features]
        
        Returns:
            Class logits [B, num_classes]
        """
        # Encode image
        img_features = self.image_encoder(image)
        
        # Encode clinical features if available
        if self.use_clinical and clinical_features is not None:
            clinical_encoded = self.clinical_encoder(clinical_features)
            # Concatenate features
            combined = torch.cat([img_features, clinical_encoded], dim=1)
        else:
            combined = img_features
        
        # Fusion and classification
        fused = self.fusion(combined)
        logits = self.classifier(fused)
        
        return logits
    
    def predict_proba(
        self,
        image: torch.Tensor,
        clinical_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(image, clinical_features)
        return F.softmax(logits, dim=1)


class SimpleImageClassifier(nn.Module):
    """
    Simplified image-only classifier for when clinical features aren't available.
    Uses ResNet-18 backbone for efficiency.
    """
    
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super(SimpleImageClassifier, self).__init__()
        
        # Load pretrained ResNet-18
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        
        # Modify first conv layer for grayscale compatibility (optional)
        # self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace classifier
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self.num_classes = num_classes
        
        # For Grad-CAM
        self.gradients = None
        self.activations = None
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


def create_model(
    model_type: str = "simple",
    num_classes: int = 4,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create model.
    
    Args:
        model_type: "simple" for image-only, "multimodal" for image+clinical
        num_classes: Number of classification classes
        pretrained: Use pretrained weights
    
    Returns:
        Model instance
    """
    if model_type == "simple":
        return SimpleImageClassifier(num_classes=num_classes, pretrained=pretrained)
    elif model_type == "multimodal":
        return AlzheimerClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_model(
    model_path: str,
    model_type: str = "simple",
    num_classes: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> nn.Module:
    """Load a trained model from disk."""
    model = create_model(model_type=model_type, num_classes=num_classes, pretrained=False)
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✅ Model loaded from {model_path}")
    else:
        print(f"⚠️ Model file not found: {model_path}. Using untrained model.")
    
    model = model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    # Simple model
    simple_model = create_model("simple", num_classes=4)
    dummy_img = torch.randn(2, 3, 224, 224)
    output = simple_model(dummy_img)
    print(f"Simple model output shape: {output.shape}")
    
    # Multi-modal model
    mm_model = create_model("multimodal", num_classes=4, num_clinical_features=8)
    dummy_clinical = torch.randn(2, 8)
    output = mm_model(dummy_img, dummy_clinical)
    print(f"Multi-modal model output shape: {output.shape}")
    
    print("✅ All tests passed!")
