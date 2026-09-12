"""FastAPI inference service for cropguard."""
from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import io
import logging
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image as PILImage
from pydantic import BaseModel, Field

from ..config import Config, load_config
from ..data.metadata import (
    WEATHER_KEYS,
    WeatherVector,
    region_index,
    stage_index,
    synthesize_weather,
)
from ..data.transforms import eval_transform
from ..hardware import get_device_info
from ..taxonomy import CROPS, GROWTH_STAGES, REGIONS, TAXONOMY
from ..training.train import build_model

log = logging.getLogger(__name__)

__all__ = ["create_app", "main"]

# --------------------------------------------------------------------------- #
# Pydantic request/response models
# --------------------------------------------------------------------------- #


class WeatherInput(BaseModel):
    temperature_c: float
    humidity_pct: float
    rainfall_mm_24h: float
    rainfall_mm_7d: float


class PredictRequest(BaseModel):
    """image_bytes accepts raw bytes (multipart) or base64 str (JSON)."""

    image_bytes: bytes | str = Field(..., description="Raw image bytes or base64 string")
    crop_id: int | None = Field(default=None, ge=1, le=len(CROPS))
    stage: str | None = None
    region: str | None = None
    month: int | None = Field(default=None, ge=1, le=12)
    weather: WeatherInput | None = None
    context_hash: str | None = None
    type_hint: str | None = Field(default=None, pattern="^(disease|pest)$")


class PredictResponse(BaseModel):
    class_name: str
    confidence: float
    severity: str
    severity_score: float
    abstain: bool
    # Why this case escalated (None when auto-resolved): 'novel_presentation',
    # 'rule_high_risk', or 'low_confidence' — officers see WHICH of the three.
    escalation_reason: str | None = None
    # Pest-detector path (NFR4 adapter): per-instance boxes in pixel coords.
    detections: list[dict[str, Any]] = Field(default_factory=list)
    # Issue 4/5: rule-layer and cold-start notes always shown to the farmer,
    # independent of whether the result was escalated.
    advisories: list[str] = Field(default_factory=list)
    probabilities: dict[str, float]
    pest_disease_id: int
    context: dict[str, Any]
    image_size: tuple[int, int]
    causal_analysis: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    config: dict[str, Any] | None


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #


def _resolve_device(cfg: Config) -> torch.device:
    """Pick the inference device (CUDA when available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_checkpoint(cfg: Config) -> Path:
    """Resolve checkpoint path from config (explicit > best > latest)."""
    if cfg.inference.checkpoint:
        return Path(cfg.inference.checkpoint)
    best = cfg.training.out_dir / "best_model.pth"
    if best.exists():
        return best
    candidates = sorted(cfg.training.out_dir.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {cfg.training.out_dir}")
    return candidates[-1]


def _load_model(cfg: Config, device: torch.device) -> torch.nn.Module:
    """Load the model from the best checkpoint."""
    checkpoint_path = _resolve_checkpoint(cfg)
    log.info(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Architecture comes from the checkpoint's own cfg — weights must match
    # what they were trained with, not the serving process's config.
    from ..training.train import build_model_from_checkpoint

    return build_model_from_checkpoint(ckpt, device, cfg)


def _resolve_temperature(cfg: Config, checkpoint_path: Path) -> float:
    """Calibration temperature (MVP-6): config override > checkpoint-fitted T > 1.0."""
    if cfg.eval.temperature is not None:
        return cfg.eval.temperature
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        t = float(ckpt.get("calibration_temperature", 1.0))
    except Exception as e:
        log.warning("Could not read calibration temperature from checkpoint: %s", e)
        t = 1.0
    return t if t > 0 else 1.0


def _load_novelty(cfg: Config, checkpoint_path: Path):
    """Novelty detector from the checkpoint; None when not fitted/enabled."""
    from .novelty import NoveltyDetector

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return NoveltyDetector.from_checkpoint(ckpt, cfg)
    except Exception as e:
        log.warning("Novelty detector unavailable: %s", e)
        return None


def create_app(cfg: Config | None = None) -> FastAPI:
    """Create the FastAPI application."""
    from ..models.pest_detector import build_pest_detector
    from .weather import WeatherService, make_redis_from_env
    from .rules import CausalRuleLayer
    from .engine import ModelRouter, SeverityEngine, temperature_scale
    from .stubs import fetch_regional_history

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        """Load the model (and calibration T) at startup; nothing to tear down."""
        try:
            model = _load_model(app.state.config, app.state.device)
            app.state.model = model
            # NFR4: disease classifier + pest head (adapter or None) behind one
            # router — the dual-model contract from Issue 2.
            app.state.router = ModelRouter(model, app.state.pest_model)
            app.state.temperature = _resolve_temperature(
                app.state.config, _resolve_checkpoint(app.state.config)
            )
            app.state.novelty = _load_novelty(
                app.state.config, _resolve_checkpoint(app.state.config)
            )
            app.state.ready = True
            log.info("Model loaded and ready")
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            app.state.ready = False
        yield
        # Shutdown: state is process-scoped; no explicit teardown needed.

    app = FastAPI(
        title="CropGuard",
        description="Causal, abstention-aware crop disease & pest diagnosis",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # State
    app.state.config = cfg or Config()
    app.state.device = _resolve_device(app.state.config)
    app.state.model = None
    app.state.router = None
    # FR3/NFR2: provider from config, shared cache from REDIS_URL (both no-ops
    # offline — fetch falls back to synthesis when the API is unreachable).
    app.state.weather_svc = WeatherService(
        redis_client=make_redis_from_env(),
        weather_cfg=app.state.config.weather,
    )
    app.state.rules = CausalRuleLayer()
    app.state.severity = SeverityEngine(app.state.config)
    app.state.temperature = app.state.config.eval.temperature or 1.0
    app.state.novelty = None  # loaded at startup from the checkpoint
    # §1 pest head (NFR4 adapter). Built here so failures surface at boot.
    app.state.pest_model = build_pest_detector(app.state.config)
    # Demo seam for FR8/Issue 5: backend detections rows land here until the
    # detections table exists. In-memory, session-scoped.
    app.state.detections: list = []
    app.state.ready = False

    # ------------------------------------------------------------------ #
    # Endpoints
    # ------------------------------------------------------------------ #

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok" if app.state.ready else "loading",
            model_loaded=app.state.ready,
            config=app.state.config.to_dict() if app.state.ready else None,
        )

    @app.get("/model-info")
    def model_info() -> dict[str, Any]:
        """Hardware + label-space report for the ops dashboard."""
        return {
            "hardware": get_device_info(),
            "num_classes": len(TAXONOMY.classes),
            "classes": TAXONOMY.classes,
            "crops": CROPS,
            "regions": REGIONS,
            "growth_stages": GROWTH_STAGES,
            "model_loaded": app.state.ready,
            "checkpoint": str(_resolve_checkpoint(app.state.config))
            if app.state.ready
            else None,
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        if not app.state.ready:
            raise HTTPException(status_code=503, detail="Model not loaded")

        router: ModelRouter = app.state.router
        cfg: Config = app.state.config
        device: torch.device = app.state.device
        weather_svc: WeatherService = app.state.weather_svc
        rules: CausalRuleLayer = app.state.rules
        severity_engine: SeverityEngine = app.state.severity

        # Validate image size
        if len(request.image_bytes) > cfg.inference.max_image_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds {cfg.inference.max_image_bytes} bytes",
            )

        # Decode image
        image_raw = request.image_bytes
        if isinstance(image_raw, str):
            try:
                image_raw = base64.b64decode(image_raw, validate=True)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}") from e
        try:
            pil_image = PILImage.open(io.BytesIO(image_raw)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}") from e

        transform = eval_transform(cfg.data.image_size)
        image_tensor = transform(pil_image).unsqueeze(0).to(device)

        # Context & Weather
        region = request.region or "pune"
        stage = request.stage or "vegetative"
        crop_id = request.crop_id if request.crop_id is not None else 1
        
        if request.weather is None:
            # FR3: Fetch and cache weather
            weather_vec = weather_svc.get_weather(region, month=request.month)
        else:
            weather_vec = WeatherVector(**request.weather.model_dump())

        crop_t = torch.tensor([crop_id], dtype=torch.long).to(device)
        stage_t = torch.tensor([stage_index(stage)], dtype=torch.long).to(device)
        region_t = torch.tensor([region_index(region)], dtype=torch.long).to(device)
        weather_t = weather_vec.to_tensor().unsqueeze(0).to(device)

        with torch.no_grad():
            output = router.predict(
                image_tensor,
                type_hint=request.type_hint,
                crop_id=crop_t,
                stage_idx=stage_t,
                region_idx=region_t,
                weather=weather_t,
            )

        # NFR4: the router may return classifier output (38-class logits) OR
        # detector output (detections list) — handle both tails explicitly.
        detections_out = output.get("detections")
        if detections_out is not None:
            top = max(detections_out, key=lambda d: d.confidence) if detections_out else None
            class_name = top.class_name if top else "no_detection"
            confidence = top.confidence if top else 0.0
            probs = None
            # Detector classes live outside the PlantVillage taxonomy; 0 =
            # "unclassified pest finding" until the backend adds their rows.
            pest_disease_id = 0
        else:
            # MVP-6: temperature-scaled confidence (T fitted at train time).
            probs = torch.softmax(
                temperature_scale(output["logits"], temperature=app.state.temperature), dim=-1
            )[0]
            max_prob, predicted = probs.max(0)
            class_name = TAXONOMY.classes[predicted.item()]
            confidence = max_prob.item()
            pest_disease_id = TAXONOMY.pest_disease_id(class_name)

        # FR5: Causal rule layer
        rule_res = rules.evaluate(crop_id, weather_vec, stage, class_name)

        # Issue 3: severity via the shared normalization seam. The pest path
        # supplies bbox coverage as the raw signal; the disease path still uses
        # calibrated confidence (Grad-CAM lesion area lands in V2).
        raw_severity = output.get("raw_severity")
        if raw_severity is not None:
            severity_tier, severity_score = severity_engine.calculate(
                class_name, raw_severity=raw_severity
            )
        else:
            severity_tier, severity_score = severity_engine.calculate(class_name, confidence)

        # FR6: Escalation & Abstention logic
        # Rule layer high risk flags can force escalation even if model is confident
        force_escalate = rule_res.risk_tier == "high"

        # Issue 5: regional history cold-start. Zero history means "unknown",
        # never "no risk" — surface it instead of letting absence read as safety.
        history_count = fetch_regional_history(app.state.detections, region, pest_disease_id)
        cold_start = history_count == 0

        # Issue 4: rule-vs-model precedence — a flagged rule layer always attaches
        # its advisory to the farmer result, even when the model is confident and
        # no escalation happens. This is what keeps early-stage weather-driven
        # risk visible before the vision model has anything to see.
        advisories: list[str] = []
        if rule_res.flagged:
            advisory = f"Rule layer advisory ({rule_res.risk_tier} risk): {rule_res.reason}"
            if rule_res.contributing_factors:
                advisory += " — " + "; ".join(sorted(rule_res.contributing_factors))
            advisories.append(advisory)
        if cold_start and not TAXONOMY.is_healthy(class_name):
            advisories.append(
                f"No regional detection history for {region} in the last 7 days — "
                "assessment based on image and weather only"
            )

        # §2: novelty check — embedding-space distance from every known-class
        # cluster. A genuinely different escalation reason from low confidence.
        is_novel = False
        if app.state.novelty is not None and output.get("features") is not None:
            try:
                is_novel = app.state.novelty.is_novel(output["features"][0].cpu())
            except Exception as e:
                log.warning("Novelty check failed (treating as in-distribution): %s", e)

        abstain = (
            (cfg.eval.confidence_threshold is not None and confidence < cfg.eval.confidence_threshold)
            or force_escalate
            or is_novel
        )
        # Precedence when several triggers fire: novel beats rule beats low-conf,
        # because "matches nothing we know" is the most actionable information
        # for the officer reviewing the queue.
        escalation_reason = (
            "novel_presentation"
            if is_novel
            else "rule_high_risk"
            if force_escalate
            else "low_confidence"
            if abstain
            else None
        )
        if is_novel:
            advisories.append(
                "Image does not match any known class in the model — routed to an officer for review"
            )

        return PredictResponse(
            class_name=class_name,
            confidence=round(confidence, 4),
            severity=severity_tier,
            severity_score=round(severity_score, 4),
            abstain=bool(abstain),
            escalation_reason=escalation_reason,
            advisories=advisories,
            probabilities={
                TAXONOMY.classes[i]: round(p.item(), 4)
                for i, p in enumerate(probs)
            }
            if probs is not None
            else {},
            pest_disease_id=pest_disease_id,
            detections=[
                {
                    "class_name": d.class_name,
                    "confidence": round(d.confidence, 4),
                    "bbox_xyxy": [round(v, 1) for v in d.xyxy],
                }
                for d in (detections_out or [])
            ],
            context={
                "crop_id": crop_id,
                "stage": stage,
                "region": region,
                "weather": weather_vec.to_dict(),
                "weather_source": weather_svc.last_source,
            },
            image_size=pil_image.size,
            causal_analysis={
                "risk_tier": rule_res.risk_tier,
                "flagged": rule_res.flagged,
                "reason": rule_res.reason,
                "contributing_factors": rule_res.contributing_factors,
                "force_escalate": force_escalate,
                "history_count": history_count,
                "cold_start": cold_start,
                "calibration_temperature": app.state.temperature,
                "escalation_reason": escalation_reason,
            }
        )

    @app.get("/config")
    def get_config() -> dict[str, Any]:
        return app.state.config.to_dict()

    @app.post("/predict/file")
    async def predict_file(
        file: UploadFile = File(...),
        region: str | None = None,
        stage: str | None = None,
        month: int | None = None,
        crop_id: int | None = None,
    ) -> PredictResponse:
        """Predict from an uploaded multipart file."""
        if file.content_type not in {
            "image/jpeg",
            "image/png",
            "image/bmp",
            "image/webp",
        }:
            raise HTTPException(status_code=400, detail="Unsupported image type")

        contents = await file.read()
        if len(contents) > app.state.config.inference.max_image_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds {app.state.config.inference.max_image_bytes} bytes",
            )
        return predict(
            PredictRequest(
                image_bytes=contents,
                crop_id=crop_id,
                stage=stage,
                region=region,
                month=month,
                context_hash=hashlib.sha256(contents).hexdigest(),
            )
        )

    return app


def main() -> None:
    """Run the FastAPI service."""
    import uvicorn

    parser = argparse.ArgumentParser(description="CropGuard inference service")
    parser.add_argument(
        "--config", type=Path, help="Path to config YAML file"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host"
    )
    parser.add_argument("--port", type=int, default=8100, help="Port")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    cfg = load_config(args.config)
    app = create_app(cfg)

    log.info(f"Starting CropGuard on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()