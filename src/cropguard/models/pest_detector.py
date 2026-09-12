"""Pest-detection head (§1): YOLOv8n/s behind the NFR4 adapter contract.

A detector returns boxes + scores + class ids, not class logits — a different
output shape than the disease classifier. `PestDetectionAdapter` maps the
detector output ONTO the router's contract (callable: model(image, **ctx) →
dict with "logits", "probs", "features" plus detector-specific extras — the
same nn.Module calling convention the disease classifier follows) so
`ModelRouter` can hold a disease classifier and a pest detector behind one
interface and pick between them by confidence, unchanged.

Two implementations:
  - UltralyticsDetector: real YOLOv8n/s via the `ultralytics` package (lazy
    import) — the production path once trained pest weights exist (Inquiry 5).
  - HeuristicSpotDetector: deterministic dark-spot blob detector, documented
    stand-in that exercises the full bbox pipeline offline (tests, demos).
    ponytail: a color-heuristic stand-in, NOT a pest model — it detects dark
    spots, not insects. Replace via model.pest_detector=yolo + weights.

bbox coverage (detected area / image area) flows out as `raw_severity`, the
real signal the Issue-3 severity seam was built to receive.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """One detected instance, pixel coordinates on the source image."""

    class_id: int
    class_name: str
    confidence: float
    # Pixel bbox (x1, y1, x2, y2).
    xyxy: tuple[float, float, float, float]

    @property
    def area(self) -> float:
        return max(0.0, self.xyxy[2] - self.xyxy[0]) * max(0.0, self.xyxy[3] - self.xyxy[1])


def denormalize_boxes(boxes_norm: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """(N, 4) xyxy in [0, 1] → pixel coordinates, clamped to the image."""
    scale = torch.tensor([width, height, width, height], dtype=boxes_norm.dtype)
    px = boxes_norm * scale
    px[:, [0, 2]] = px[:, [0, 2]].clamp(0, width)
    px[:, [1, 3]] = px[:, [1, 3]].clamp(0, height)
    return px


def bbox_coverage(detections: list[Detection], width: int, height: int) -> float:
    """Union bbox area / image area in [0, 1] — the Issue-3 raw severity signal.

    ponytail: union-of-independent-boxes (overlaps counted twice); swap for
    pixel-accurate mask union when segmentation lands (V2 plan).
    """
    if not detections or width <= 0 or height <= 0:
        return 0.0
    total = sum(d.area for d in detections)
    return float(min(1.0, total / (width * height)))


class PestDetectionAdapter:
    """Maps detector output onto the router contract (NFR4).

    Output dict carries the router-contract keys plus:
      detections: list[Detection]   (pixel coords)
      raw_severity: float           (bbox coverage in [0, 1])
      image_size: (width, height)
    """

    def __init__(self, detector) -> None:
        self.detector = detector

    def predict(self, image: torch.Tensor, type_hint: str | None = None, **context) -> dict:
        # Router passes a normalized batched tensor; detectors want a PIL image.
        img = self._to_pil(image)
        width, height = img.size
        dets = self.detector.detect(img)
        top = max(dets, key=lambda d: d.confidence) if dets else None

        return {
            "logits": self._confidence_logits(top.confidence if top else 0.0),
            "probs": None,  # detectors have no class distribution; router uses max-prob only
            "features": None,
            "detections": dets,
            "raw_severity": bbox_coverage(dets, width, height),
            "image_size": (width, height),
        }

    def _to_pil(self, image: torch.Tensor) -> Image.Image:
        from ..data.transforms import IMAGENET_MEAN, IMAGENET_STD

        t = image[0] if image.dim() == 4 else image  # (B,3,H,W) → (3,H,W)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        arr = ((t.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)

    # Router invokes models as callables (nn.Module convention); this alias is
    # the whole adapter-to-router seam.
    __call__ = predict

    @staticmethod
    def _confidence_logits(confidence: float) -> torch.Tensor:
        """Single-column pseudo-logit encoding the top detection's confidence:
        sigmoid(logits).max() == confidence exactly, so the router's
        confidence-based routing works unchanged across the full [0,1] range."""
        p = min(max(confidence, 1e-6), 1.0 - 1e-6)
        logit = float(np.log(p / (1.0 - p)))
        return torch.tensor([[logit]])


class UltralyticsDetector:
    """Real YOLOv8n/s via ultralytics. Requires inference.pest_weights."""

    def __init__(self, weights: Path, conf_threshold: float = 0.25) -> None:
        try:
            from ultralytics import YOLO  # lazy: heavy dep, only needed for this head
        except ImportError as e:
            raise ImportError(
                "model.pest_detector=yolo requires the `ultralytics` package (pip install ultralytics)"
            ) from e
        self.model = YOLO(str(weights))
        self.conf = conf_threshold

    def detect(self, image: Image.Image) -> list[Detection]:
        results = self.model.predict(image, conf=self.conf, verbose=False)
        dets: list[Detection] = []
        for res in results:
            names = res.names
            for box in res.boxes:
                cls_id = int(box.cls.item())
                dets.append(
                    Detection(
                        class_id=cls_id,
                        class_name=str(names.get(cls_id, cls_id)),
                        confidence=float(box.conf.item()),
                        xyxy=tuple(float(v) for v in box.xyxy[0].tolist()),
                    )
                )
        return dets


class HeuristicSpotDetector:
    """Deterministic dark-spot detector — offline stand-in for the pest head.

    ponytail: detects dark blobs (candidate damage sites), not insects. Exists
    so the bbox pipeline, severity wiring, and router integration are testable
    before pest weights exist. Upgrade path: model.pest_detector=yolo.
    """

    def __init__(self, sensitivity: float = 0.55) -> None:
        self.sensitivity = sensitivity

    def detect(self, image: Image.Image) -> list[Detection]:
        gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        mask = gray < self.sensitivity * gray.mean()  # dark spots vs leaf tone

        from scipy import ndimage  # numpy-adjacent; part of the scientific stack

        labeled, n = ndimage.label(mask)
        dets: list[Detection] = []
        for i in range(1, n + 1):
            ys, xs = np.where(labeled == i)
            area_px = len(xs)
            if area_px < 25:  # noise floor: skip specks
                continue
            x1, x2 = float(xs.min()), float(xs.max()) + 1.0
            y1, y2 = float(ys.min()), float(ys.max()) + 1.0
            fill = area_px / max(1.0, (x2 - x1) * (y2 - y1))
            conf = float(min(1.0, fill * 1.2))  # compact blobs read more pest-like
            dets.append(
                Detection(
                    class_id=0,
                    class_name="spot_damage",
                    confidence=round(conf, 3),
                    xyxy=(x1, y1, x2, y2),
                )
            )
        return sorted(dets, key=lambda d: d.confidence, reverse=True)


def build_pest_detector(cfg) -> PestDetectionAdapter | None:
    """Factory from config. None = no pest head (router runs disease-only)."""
    mode = cfg.model.pest_detector
    if mode == "off":
        return None
    if mode == "heuristic":
        return PestDetectionAdapter(HeuristicSpotDetector())
    if mode == "yolo":
        if not cfg.inference.pest_weights:
            raise ValueError("model.pest_detector=yolo requires inference.pest_weights")
        return PestDetectionAdapter(UltralyticsDetector(cfg.inference.pest_weights))
    raise ValueError(f"Unknown pest_detector: {mode}")
