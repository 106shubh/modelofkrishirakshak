"""Single-image prediction from the command line.

Loads the best (or given) checkpoint, validates the image, synthesizes
deterministic context metadata when the caller does not supply any, and prints
a diagnosis with confidence, severity band, and abstention flag.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torchvision import transforms  # noqa: F401  (kept for API parity)

from ..config import Config, load_config
from ..data.dataset import digest_of
from ..data.metadata import (
    WEATHER_KEYS,
    WeatherVector,
    region_index,
    stage_index,
    synthesize_weather,
)
from ..data.transforms import eval_transform
from ..data.validation import load_validated_image
from ..taxonomy import CROPS, GROWTH_STAGES, REGIONS, TAXONOMY
from ..training.train import build_model

log = logging.getLogger(__name__)

MONTHS = list(range(1, 13))


def severity_band(confidence: float, bands: list[float]) -> str:
    """Map a confidence value onto configured severity bands."""
    if confidence < bands[0]:
        return "low"
    if confidence < bands[1]:
        return "medium"
    return "high"


def predict(
    model: torch.nn.Module,
    image_path: Path,
    crop_id: int | None,
    stage: str | None,
    region: str | None,
    month: int | None,
    weather: dict[str, float] | None,
    device: torch.device,
    cfg: Config,
) -> dict:
    """Run a single prediction."""
    model.eval()

    result = load_validated_image(image_path)
    if result is None:
        return {"error": "invalid image"}

    pil_image, (w, h), sha = result
    transform = eval_transform(cfg.data.image_size)
    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Deterministic context synthesis when not provided (seeded by file digest).
    digest = digest_of(str(image_path))
    if crop_id is None:
        crop_id = 1 + digest % len(CROPS)
    if stage is None:
        stage = GROWTH_STAGES[digest % len(GROWTH_STAGES)]
    if region is None:
        region = REGIONS[digest % len(REGIONS)]
    if month is None:
        month = MONTHS[digest % 12]

    crop_t = torch.tensor([crop_id], dtype=torch.long).to(device)
    stage_t = torch.tensor([stage_index(stage)], dtype=torch.long).to(device)
    region_t = torch.tensor([region_index(region)], dtype=torch.long).to(device)

    if weather is None:
        weather_vec = synthesize_weather(region, month, str(image_path))
    else:
        weather_vec = WeatherVector(**{k: weather.get(k, 0.0) for k in WEATHER_KEYS})
    weather_t = weather_vec.to_tensor().unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(
            image_tensor,
            crop_id=crop_t,
            stage_idx=stage_t,
            region_idx=region_t,
            weather=weather_t,
        )
        probs = output["probs"][0]

    # Abstention by confidence threshold.
    max_prob, predicted = probs.max(0)
    class_name = TAXONOMY.classes[predicted.item()]
    confidence = max_prob.item()
    abstain = (
        cfg.eval.confidence_threshold is not None
        and confidence < cfg.eval.confidence_threshold
    )

    return {
        "class_name": class_name,
        "confidence": round(confidence, 4),
        "abstain": bool(abstain),
        "severity": severity_band(confidence, cfg.eval.severity_bands),
        "pest_disease_id": TAXONOMY.pest_disease_id(class_name),
        "probabilities": {
            TAXONOMY.classes[i]: round(p.item(), 4) for i, p in enumerate(probs)
        },
        "context": {
            "crop_id": crop_id,
            "stage": stage,
            "region": region,
            "month": month,
            "weather": weather_vec.to_dict(),
        },
        "image_size": (w, h),
        "sha256": sha,
    }


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Predict crop disease")
    parser.add_argument("image", type=Path, help="Path to input image")
    parser.add_argument("--config", type=Path, help="Path to config YAML file")
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="Path to model checkpoint"
    )
    parser.add_argument("--crop-id", type=int, default=None, help="Crop ID override")
    parser.add_argument("--stage", type=str, default=None, help="Growth stage override")
    parser.add_argument("--region", type=str, default=None, help="Region override")
    parser.add_argument("--month", type=int, default=None, help="Month override")
    parser.add_argument(
        "--set",
        nargs="*",
        dest="overrides",
        help="Config overrides, e.g. --set data.image_size=160",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    cfg = load_config(args.config, overrides=args.overrides) if args.config else load_config(None, overrides=args.overrides)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint = args.checkpoint or cfg.training.out_dir / "best_model.pth"
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        log.error(f"Checkpoint not found: {checkpoint}")
        raise SystemExit(1)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    from .train import build_model_from_checkpoint

    model = build_model_from_checkpoint(ckpt, device, cfg)

    result = predict(
        model=model,
        image_path=args.image,
        crop_id=args.crop_id,
        stage=args.stage,
        region=args.region,
        month=args.month,
        weather=None,
        device=device,
        cfg=cfg,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    elif result.get("error"):
        log.error(result["error"])
        raise SystemExit(1)
    else:
        log.info(f"Class: {result['class_name']}")
        log.info(f"Confidence: {result['confidence']:.4f}")
        log.info(f"Severity: {result['severity']}")
        log.info(f"Abstain: {result['abstain']}")
        log.info(f"Context: {result['context']}")


if __name__ == "__main__":
    main()
