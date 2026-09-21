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
    # --- New Structured Schema ---
    crop: str = ""
    diagnosis: dict[str, Any] = Field(default_factory=dict)
    image_quality: dict[str, Any] = Field(default_factory=dict)
    severity_info: dict[str, Any] = Field(default_factory=dict) 
    explainability: dict[str, Any] = Field(default_factory=dict)
    environmental_risk: dict[str, Any] = Field(default_factory=dict)
    regional_context: dict[str, Any] = Field(default_factory=dict)
    recommendation: dict[str, Any] = Field(default_factory=dict)
    abstention: dict[str, Any] = Field(default_factory=dict)

    # --- Backward Compatibility ---
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
    from .quality import assess_image_quality
    from .explain import explain_prediction

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
        try:
            return _predict_inner(request)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
            
    def _predict_inner(request: PredictRequest) -> PredictResponse:
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
            
        # [NEW] Image Quality Check
        quality_res = assess_image_quality(pil_image)

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
        
        model_kwargs = {
            "type_hint": request.type_hint,
            "crop_id": crop_t,
            "stage_idx": stage_t,
            "region_idx": region_t,
            "weather": weather_t
        }

        with torch.no_grad():
            output = router.predict(image_tensor, **model_kwargs)

        # Variables for the new response format
        margin = 0.0
        top_alternatives = []
        is_novel = False
        activated_ratio = None
        predicted_idx = 0
        probs_dict = {}

        # NFR4: the router may return classifier output (38-class logits) OR
        # detector output (detections list) — handle both tails explicitly.
        detections_out = output.get("detections")
        if detections_out is not None:
            top = max(detections_out, key=lambda d: d.confidence) if detections_out else None
            class_name = top.class_name if top else "no_detection"
            confidence = top.confidence if top else 0.0
            probs = None
            pest_disease_id = 0
        else:
            # MVP-6: temperature-scaled confidence (T fitted at train time).
            probs = torch.softmax(
                temperature_scale(output["logits"], temperature=app.state.temperature), dim=-1
            )[0]
            max_prob, predicted = probs.max(0)
            predicted_idx = predicted.item()
            class_name = TAXONOMY.classes[predicted_idx]
            confidence = max_prob.item()
            pest_disease_id = TAXONOMY.pest_disease_id(class_name)
            
            # [NEW] Calculate margin and top alternatives
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            if len(sorted_probs) > 1:
                margin = (sorted_probs[0] - sorted_probs[1]).item()
                
            for i in range(1, min(4, len(sorted_probs))):
                alt_class_name = TAXONOMY.classes[sorted_indices[i].item()]
                top_alternatives.append({
                    "disease": alt_class_name,
                    "probability": round(sorted_probs[i].item(), 4)
                })
                
            probs_dict = {
                TAXONOMY.classes[i]: round(p.item(), 4)
                for i, p in enumerate(probs)
            }
            
        # [NEW] Explainability via Grad-CAM
        explainability = {"method": "None", "available": False, "reason": "No detector active."}
        if detections_out is None and class_name != "no_detection":
            explainability = explain_prediction(
                router, image_tensor, pil_image, predicted_idx, **model_kwargs
            )
            activated_ratio = explainability.get("activated_ratio")
            if "activated_ratio" in explainability:
                del explainability["activated_ratio"]

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
            severity_basis = ["Calculated from object detection bounding box coverage"]
        elif activated_ratio is not None:
            # [NEW] Decouple disease severity from confidence using Grad-CAM activation
            severity_tier, severity_score = severity_engine.calculate(
                class_name, raw_severity=activated_ratio
            )
            severity_basis = ["Calculated from Grad-CAM activation heatmap area"]
        else:
            severity_tier, severity_score = severity_engine.calculate(class_name, confidence)
            severity_basis = ["Calculated using confidence proxy (requires visual assessment)"]

        # FR6: Escalation & Abstention logic
        # Rule layer high risk flags can force escalation even if model is confident
        force_escalate = rule_res.risk_tier == "high"

        # Issue 5: regional history cold-start. Zero history means "unknown",
        # never "no risk" — surface it instead of letting absence read as safety.
        history_count = fetch_regional_history(app.state.detections, region, pest_disease_id)
        cold_start = history_count == 0

        # §2: novelty check — embedding-space distance from every known-class
        # cluster. A genuinely different escalation reason from low confidence.
        if app.state.novelty is not None and output.get("features") is not None:
            try:
                is_novel = app.state.novelty.is_novel(output["features"][0].cpu())
            except Exception as e:
                log.warning("Novelty check failed (treating as in-distribution): %s", e)

        # [NEW] Abstention logic incorporates margin and image quality
        min_conf = getattr(cfg.eval, "min_confidence", 0.65)
        min_margin = getattr(cfg.eval, "min_margin", 0.10)
        
        is_low_conf = (confidence < min_conf) or (margin < min_margin)
        is_poor_quality = quality_res["status"] == "poor"

        abstain = is_low_conf or force_escalate or is_novel or is_poor_quality
        
        # Precedence when several triggers fire
        if is_poor_quality:
            escalation_reason = "poor_image_quality"
        elif is_novel:
            escalation_reason = "novel_presentation"
        elif force_escalate:
            escalation_reason = "rule_high_risk"
        elif abstain:
            escalation_reason = "low_confidence_or_margin"
        else:
            escalation_reason = None
            
        # Issue 4: rule-vs-model precedence — a flagged rule layer always attaches
        # its advisory to the farmer result, even when the model is confident and
        # no escalation happens.
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
        if is_novel:
            advisories.append(
                "Image does not match any known class in the model — routed to an officer for review"
            )

        # [NEW] Construct detailed JSON blocks
        crop_name = CROPS.get(crop_id, "Unknown").capitalize()
        
        disease_name = class_name.split("___")[-1].replace("_", " ") if "___" in class_name else class_name
        diagnosis = {
            "disease": disease_name,
            "confidence": round(confidence, 4),
            "confidence_percent": round(confidence * 100, 2),
            "top_alternatives": top_alternatives,
            "margin": round(margin, 4),
            "reliable": not (is_low_conf or is_novel)
        }
        
        severity_info = {
            "level": severity_tier,
            "score": round(severity_score, 4),
            "basis": severity_basis,
            "confidence": "moderate" if activated_ratio is not None else "low"
        }
        
        environmental_risk = {
            "tier": rule_res.risk_tier,
            "factors": rule_res.contributing_factors,
            "interpretation": rule_res.reason if rule_res.flagged else "Environmental conditions do not pose a severe risk."
        }
        
        regional_context = {
            "available": not cold_start,
            "history_count": history_count,
            "message": f"Recorded {history_count} occurrences in {region} over the past 7 days." if not cold_start else f"No regional detection history available for {region}."
        }
        
        recommendation = {
            "status": "inspection_required" if abstain else "advisory",
            "actions": [
                "Schedule a field inspection or consult a local agricultural officer." if abstain else "Monitor the crop closely and consider preventive IPM strategies.",
                "Avoid aggressive chemical treatment until diagnosis is confirmed visually." if abstain else f"Review regional guidelines for treating {disease_name}."
            ]
        }
        
        abstention_details = {
            "required": bool(abstain),
            "reason": escalation_reason
        }

        return PredictResponse(
            crop=crop_name,
            diagnosis=diagnosis,
            image_quality=quality_res,
            severity_info=severity_info,
            explainability=explainability,
            environmental_risk=environmental_risk,
            regional_context=regional_context,
            recommendation=recommendation,
            abstention=abstention_details,
            
            # Backwards compatibility fields
            class_name=class_name,
            confidence=round(confidence, 4),
            severity=severity_tier,
            severity_score=round(severity_score, 4),
            abstain=bool(abstain),
            escalation_reason=escalation_reason,
            advisories=advisories,
            probabilities=probs_dict,
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
        from pydantic import ValidationError
        try:
            req = PredictRequest(
                image_bytes=contents,
                crop_id=crop_id,
                stage=stage,
                region=region,
                month=month,
                context_hash=hashlib.sha256(contents).hexdigest(),
            )
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
            
        return predict(req)

    return app

# Expose global ASGI app for deployment via uvicorn
try:
    from pathlib import Path
    from ..config import load_config
    _demo_cfg = load_config(Path("configs/demo.yaml"))
except Exception as e:
    import logging
    logging.warning(f"Could not load demo.yaml config: {e}")
    _demo_cfg = None

app = create_app(_demo_cfg)

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