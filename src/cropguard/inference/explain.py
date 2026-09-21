"""Explainability module for CropGuard.

Implements Grad-CAM for CNN backbones.
"""
from __future__ import annotations

import base64
import io
import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

log = logging.getLogger(__name__)


class GradCAM:
    """Simple Grad-CAM implementation."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        
        # Register hooks
        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def generate(self, input_tensor: torch.Tensor, class_idx: int, **kwargs) -> np.ndarray | None:
        """Generate Grad-CAM heatmap for a specific class index."""
        underlying = getattr(self.model, "disease_model", self.model)
        underlying.eval()
        underlying.zero_grad()
        
        # Forward pass
        # Since it's our multimodal router, we need to pass the context kwargs
        if hasattr(self.model, "predict"):
            output = self.model.predict(input_tensor, **kwargs)
        else:
            output = self.model(input_tensor, **kwargs)
        
        if "logits" not in output:
            return None
            
        logits = output["logits"]
        if logits.dim() > 1:
            logits = logits[0]
            
        # Backward pass
        score = logits[class_idx]
        score.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            return None
            
        # Global average pooling on gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        
        # ReLU to keep only positive influence
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-6:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
            
        return cam.cpu().numpy()


def overlay_heatmap(img: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Overlay a heatmap on the original image."""
    img_arr = np.array(img.convert("RGB"))
    
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (img_arr.shape[1], img_arr.shape[0]))
    
    # Convert to heatmap color map
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = cv2.addWeighted(img_arr, 1 - alpha, heatmap_color, alpha, 0)
    return Image.fromarray(overlay)


def encode_image_base64(img: Image.Image, format: str = "JPEG") -> str:
    """Encode PIL Image to base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=format, quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def explain_prediction(
    model: torch.nn.Module, 
    input_tensor: torch.Tensor, 
    original_image: Image.Image, 
    class_idx: int,
    **kwargs
) -> dict:
    """Wrapper to generate explainability output."""
    # Find the target layer. We assume it's a FeatureBackbone internally containing an efficientnet
    target_layer = None
    
    # Try to drill down to the actual backbone
    if hasattr(model, "disease_model"):
        net = getattr(model.disease_model, "backbone", None)
        if net and hasattr(net, "net"):
            backbone_net = net.net
            if hasattr(backbone_net, "features"):
                target_layer = backbone_net.features[-1]
                
    if target_layer is None:
        return {
            "method": "None",
            "available": False,
            "reason": "Current classifier architecture is incompatible with Grad-CAM hooking.",
            "heatmap": None
        }
        
    try:
        gradcam = GradCAM(model, target_layer)
        # We need to enable grad for the input tensor momentarily if it wasn't
        underlying_model = getattr(model, "disease_model", model)
        was_training = underlying_model.training
        underlying_model.eval()
        
        heatmap = gradcam.generate(input_tensor, class_idx, **kwargs)
        gradcam.remove_hooks()
        
        if was_training:
            underlying_model.train()
            
        if heatmap is None:
            return {
                "method": "Grad-CAM",
                "available": False,
                "reason": "Failed to compute gradients.",
                "heatmap": None
            }
            
        overlay_img = overlay_heatmap(original_image, heatmap)
        b64_str = encode_image_base64(overlay_img)
        
        # Calculate heuristic severity (activated area > 0.5)
        # This is a crude heuristic just to have some basis for severity decoupled from confidence
        activated_ratio = float(np.mean(heatmap > 0.5))
        
        return {
            "method": "Grad-CAM",
            "available": True,
            "heatmap": b64_str,
            "activated_ratio": activated_ratio
        }
    except Exception as e:
        log.warning("Grad-CAM failed: %s", e)
        return {
            "method": "Grad-CAM",
            "available": False,
            "reason": f"Execution failed: {str(e)}",
            "heatmap": None
        }
