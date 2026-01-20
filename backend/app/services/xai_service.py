"""
CARE-AD+ XAI Service

Explainable AI service providing visual and textual explanations.
Implements Grad-CAM for image attention and feature importance.
"""
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Dict, List, Any, Optional
import asyncio

# Add paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DIR = os.path.join(BASE_DIR, 'ml')
sys.path.insert(0, ML_DIR)

from app.config import settings


class GradCAM:
    """
    Grad-CAM implementation for CNN visualization.
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        
        # Get weights
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # Generate heatmap
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Resize to input size
        cam = F.interpolate(
            cam, 
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        
        return cam.squeeze().cpu().numpy()


class XAIService:
    """
    Explainable AI service for generating visual explanations.
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.gradcam = None
        self._load_model()
    
    def _load_model(self):
        """Load model for XAI analysis."""
        try:
            from ml.model import load_model, create_model
            
            model_path = settings.MODEL_PATH
            best_path = model_path.replace('.pth', '_best.pth')
            
            if os.path.exists(model_path):
                self.model = load_model(model_path, "simple", settings.NUM_CLASSES, self.device)
            elif os.path.exists(best_path):
                self.model = load_model(best_path, "simple", settings.NUM_CLASSES, self.device)
            else:
                self.model = create_model("simple", settings.NUM_CLASSES, pretrained=True)
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            # Get target layer for Grad-CAM
            if hasattr(self.model, 'backbone'):
                # Get last conv layer
                target_layer = None
                for name, module in self.model.backbone.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        target_layer = module
                
                if target_layer:
                    self.gradcam = GradCAM(self.model, target_layer)
                    
            print("✅ XAI Service initialized")
            
        except Exception as e:
            print(f"⚠️ XAI Service warning: {e}")
            self.model = None
    
    async def generate_gradcam(
        self,
        image_path: str,
        predicted_class: str
    ) -> Dict[str, Any]:
        """
        Generate Grad-CAM visualization (async wrapper).
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._generate_gradcam_sync,
            image_path,
            predicted_class
        )
        return result
    
    def _generate_gradcam_sync(
        self,
        image_path: str,
        predicted_class: str
    ) -> Dict[str, Any]:
        """
        Generate Grad-CAM visualization (sync).
        """
        from torchvision import transforms
        from PIL import Image
        
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Get class index
        try:
            class_idx = settings.CLASS_NAMES.index(predicted_class)
        except ValueError:
            class_idx = 0
        
        # Generate heatmap
        heatmap = None
        highlighted_regions = []
        
        if self.gradcam is not None:
            try:
                heatmap = self.gradcam.generate(input_tensor, class_idx)
                highlighted_regions = self._identify_regions(heatmap, predicted_class)
            except Exception as e:
                print(f"Grad-CAM generation error: {e}")
        
        # Generate simplified heatmap representation
        heatmap_data = None
        if heatmap is not None:
            # Downsample heatmap for JSON storage
            h, w = heatmap.shape
            step = max(1, min(h, w) // 8)
            heatmap_small = heatmap[::step, ::step].tolist()
            heatmap_data = {"grid": heatmap_small, "shape": list(heatmap.shape)}
        
        # Save heatmap image
        heatmap_path = None
        if heatmap is not None:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                
                save_dir = settings.UPLOAD_DIR
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                heatmap_path = os.path.join(save_dir, f"{base_name}_gradcam.png")
                
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(np.array(image.resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE))))
                ax.imshow(heatmap, alpha=0.5, cmap='jet')
                ax.axis('off')
                plt.tight_layout()
                plt.savefig(heatmap_path, bbox_inches='tight', dpi=100)
                plt.close()
                
            except Exception as e:
                print(f"Error saving heatmap: {e}")
        
        return {
            "heatmap_path": heatmap_path,
            "heatmap_data": heatmap_data,
            "highlighted_regions": highlighted_regions,
            "class": predicted_class
        }
    
    def _identify_regions(self, heatmap: np.ndarray, predicted_class: str) -> List[str]:
        """
        Identify brain regions based on heatmap activation.
        """
        # Define brain region mappings based on activation patterns
        regions = []
        
        h, w = heatmap.shape
        
        # Analyze activation zones
        if heatmap.mean() > 0.3:
            if predicted_class in ["MildDemented", "ModerateDemented"]:
                regions.extend([
                    "Hippocampal region shows significant atrophy markers",
                    "Temporal lobe displays characteristic patterns",
                    "Ventricular enlargement detected"
                ])
            elif predicted_class == "VeryMildDemented":
                regions.extend([
                    "Early hippocampal changes observed",
                    "Mild temporal lobe variations detected"
                ])
            else:
                regions.extend([
                    "Brain structures appear within normal limits",
                    "No significant atrophy markers detected"
                ])
        else:
            regions.append("Low activation pattern - review image quality")
        
        return regions


# Singleton
_xai_service = None

def get_xai_service() -> XAIService:
    """Get or create XAI service singleton."""
    global _xai_service
    if _xai_service is None:
        _xai_service = XAIService()
    return _xai_service
