"""Image quality assessment for inference.

Checks for extreme blur, darkness, or brightness before running the model.
"""
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# Thresholds
MIN_LAPLACIAN_VAR = 50.0  # Blur detection
MIN_BRIGHTNESS = 20.0     # Too dark
MAX_BRIGHTNESS = 240.0    # Too bright

def assess_image_quality(image: Image.Image) -> dict[str, Any]:
    """Assess the quality of an uploaded image.
    
    Returns:
        dict: {
            "score": float,
            "status": str ("good" or "poor"),
            "issues": list[str]
        }
    """
    # Convert PIL Image to OpenCV format (BGR or Grayscale)
    # OpenCV prefers BGR, but for these checks, Grayscale is sufficient and faster
    img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    issues: list[str] = []
    
    # Blur detection via Laplacian variance
    laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    if laplacian_var < MIN_LAPLACIAN_VAR:
        issues.append("blur")
        
    # Brightness detection
    mean_brightness = np.mean(img_gray)
    if mean_brightness < MIN_BRIGHTNESS:
        issues.append("too_dark")
    elif mean_brightness > MAX_BRIGHTNESS:
        issues.append("too_bright")
        
    # Heuristic scoring based on laplacian variance (max out at ~500)
    # and whether it's within brightness bounds.
    # This is a simple heuristic score in [0, 1].
    
    blur_score = min(1.0, max(0.0, laplacian_var / 500.0))
    brightness_score = 1.0
    if mean_brightness < MIN_BRIGHTNESS:
        brightness_score = mean_brightness / MIN_BRIGHTNESS
    elif mean_brightness > MAX_BRIGHTNESS:
        brightness_score = (255 - mean_brightness) / (255 - MAX_BRIGHTNESS)
        
    final_score = round(float((blur_score + brightness_score) / 2.0), 4)
    
    status = "poor" if issues else "good"
    
    return {
        "score": final_score,
        "status": status,
        "issues": issues
    }
