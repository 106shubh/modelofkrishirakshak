"""Hardware detection and device selection.

Detects CUDA availability and VRAM; falls back gracefully to CPU. Batch size
and precision are tuned to what is actually available — never assumed.
"""
from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_info() -> dict:
    """Return a machine-readable hardware report (for /model-info and logs)."""
    info: dict = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(get_device()),
        "cpu_threads": os.cpu_count() or 1,
        "torch_threads": torch.get_num_threads(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_vram_bytes"] = props.total_memory
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_capability"] = tuple(int(x) for x in torch.cuda.get_device_capability(0))
    else:
        info["gpu_name"] = None
        info["gpu_vram_bytes"] = None
    return info


def recommended_batch_size(cfg_batch: int, image_size: int, backbone: str) -> int:
    """Scale the configured batch size down to fit available memory.

    Heuristic based on measured per-sample VRAM on a GTX 1650 (4GB) at fp16;
    on CPU any batch size is fine so we keep the configured value.
    """
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory
        approx_per_sample = {
            "efficientnet_b0": 8e6,  # bytes @ 224px fp16 incl. activations
            "efficientnet_b1": 12e6,
            "convnext_tiny": 30e6,
            "vit_b_16": 40e6,
            "swin_t": 36e6,
        }.get(backbone, 20e6)
        scale = (vram * 0.85) / (approx_per_sample * max(cfg_batch, 1))
        fitted = max(1, int(cfg_batch * scale) // 1)
        fitted = min(fitted, cfg_batch)
        log.info("GPU VRAM=%.1fGB → batch_size %d -> %d", vram / 1e9, cfg_batch, fitted)
        return fitted
    log.info("CPU device → keeping configured batch_size=%d", cfg_batch)
    return cfg_batch


def use_amp(cfg_amp: str) -> bool:
    """Decide whether to enable mixed precision."""
    if cfg_amp == "on":
        return True
    if cfg_amp == "off":
        return False
    # auto: only on CUDA where fp16 is actually faster
    return torch.cuda.is_available()