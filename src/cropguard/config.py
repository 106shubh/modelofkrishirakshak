"""Configuration system.

All hyperparameters, thresholds, and paths live in YAML config files (or env
vars). Nothing important is hardcoded in model or API logic.

Precedence (low → high):
  1. built-in defaults
  2. config file (e.g. configs/default.yaml)
  3. environment variables (CROPGUARD_*)
  4. CLI overrides (--config.k=v style)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# Sub-configs
# --------------------------------------------------------------------------- #


class DataConfig(BaseModel):
    root: Path = PROJECT_ROOT / "data" / "raw" / "PlantVillage"
    variant: str = "color"  # color | grayscale | segmented
    image_size: int = 192
    train_subset_per_class: int | None = 300  # None = use everything
    val_subset_per_class: int | None = None  # None = use the whole val split
    val_ratio: float = 0.20  # carved deterministically from the train protocol
    workers: int = 2
    min_samples_per_class: int = 30
    duplicate_threshold: float = 0.05  # Hamming-dist fraction for perceptual dup
    # §1 acceptance criterion: lab vs field-condition accuracy must be reported
    # SEPARATELY. Point this at a PlantDoc-style/field dataset root when one
    # exists; None (or a missing dir) skips the field split.
    field_root: Path | None = None


class ModelConfig(BaseModel):
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    dropout: float = 0.2
    fusion: str = "gated"  # gated | concat | image_only
    context_dim: int = 128
    freeze_backbone: bool = False
    # §1 pest-detection head. "off" = disease model handles everything (the
    # pre-dual-model behavior); "heuristic" = offline spot detector (tests/
    # demos); "yolo" = ultralytics YOLOv8n/s via inference.pest_weights.
    pest_detector: str = "off"


class InferenceConfig(BaseModel):
    mode: str = "local"  # local | remote
    host: str = "0.0.0.0"
    port: int = 8100
    checkpoint: Path | None = None  # None → latest in out_dir
    remote_url: str = ""
    max_image_bytes: int = 15 * 1024 * 1024
    # Pest-head weights (YOLOv8 .pt). Required when model.pest_detector=yolo.
    pest_weights: Path | None = None


class TrainingConfig(BaseModel):
    batch_size: int = 32
    epochs: int = 8
    max_steps: int | None = None
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    scheduler: str = "cosine"
    early_stopping_patience: int = 3
    grad_accum: int = 1
    amp: str = "auto"  # auto | on | off
    seed: int = 42
    log_every: int = 20
    out_dir: Path = PROJECT_ROOT / "checkpoints"


class EvalConfig(BaseModel):
    confidence_threshold: float = 0.65
    ood_energy_threshold: float | None = None  # legacy knob; novelty layer uses eval.novelty
    severity_bands: list[float] = Field(default_factory=lambda: [0.34, 0.67])
    mc_dropout_samples: int = 10
    ood_abstain: bool = True
    # Confidence-calibration temperature (MVP-6). None → fitted on val logits at
    # train time and stamped into the checkpoint; a float forces a fixed value.
    temperature: float | None = None
    # Which split calibration T is fitted on. NFR5: use "test" (or a field
    # split) once field-condition data exists — never trust lab-fit T alone.
    calibration_split: str = "val"
    # §1: label smoothing for the classifier loss (0 disables).
    label_smoothing: float = 0.0
    # §2: novelty (out-of-distribution) detection over the embedding space.
    novelty: NoveltyConfig = Field(default_factory=lambda: NoveltyConfig())


class NoveltyConfig(BaseModel):
    """Embedding-space novelty detection (§2).

    enabled defaults to True so a checkpoint carrying fitted stamps serves
    WITH novelty protection out of the box; when the checkpoint has no stamps
    the detector is simply absent (from_checkpoint returns None).
    """

    enabled: bool = True
    k: int = 5  # neighbors for the k-NN distance score
    threshold: float | None = None  # None → fitted at train time, stamped into ckpt


class WeatherConfig(BaseModel):
    """Weather source (FR3). provider: open-meteo | imd | synthesize.

    open-meteo needs no credentials (10k calls/day non-commercial, CC BY 4.0
    attribution required). imd requires an approved IMD API key. The service
    falls back to seasonal synthesis whenever a fetch fails.
    """

    provider: str = "synthesize"
    api_key: str | None = None  # IMD credential; also CROPGUARD_WEATHER__API_KEY
    timeout_sec: float = 10.0
    cache_ttl_sec: int = 3600  # FR3/NFR2: per taluka+time-bucket, not per farmer


class Config(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    weather: WeatherConfig = Field(default_factory=WeatherConfig)

    @model_validator(mode="after")
    def _validate(self) -> "Config":
        if self.model.fusion not in {"gated", "concat", "image_only"}:
            raise ValueError(f"Unknown fusion: {self.model.fusion}")
        if self.data.variant not in {"color", "grayscale", "segmented"}:
            raise ValueError(f"Unknown variant: {self.data.variant}")
        if self.inference.mode not in {"local", "remote"}:
            raise ValueError(f"Unknown inference mode: {self.inference.mode}")
        if not (0.0 <= self.eval.confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be in [0, 1]")
        if self.eval.temperature is not None and self.eval.temperature <= 0.0:
            raise ValueError("eval.temperature must be > 0 when set")
        if self.weather.provider not in {"open-meteo", "imd", "synthesize"}:
            raise ValueError(f"Unknown weather provider: {self.weather.provider}")
        if self.model.pest_detector not in {"off", "heuristic", "yolo"}:
            raise ValueError(f"Unknown pest_detector: {self.model.pest_detector}")
        if self.model.pest_detector == "yolo" and not self.inference.pest_weights:
            raise ValueError("model.pest_detector=yolo requires inference.pest_weights")
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        return cls.model_validate(d)


def _apply_env(cfg: Config) -> Config:
    """Apply CROPGUARD_* environment overrides, e.g. CROPGUARD_EVAL__CONFIDENCE_THRESHOLD.

    Section and field are separated by a double underscore: CROPGUARD_<section>__<field>.
    """
    import os

    prefix = "CROPGUARD_"
    node: dict[str, Any] | None = None
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix) :].lower().split("__")
        if node is None:
            node = cfg.to_dict()
        cur = node
        try:
            for part in path[:-1]:
                cur = cur[part]
            leaf = path[-1]
            current = cur[leaf]
        except (KeyError, TypeError):
            log.warning("Ignoring unknown env override: %s", key)
            continue
        if isinstance(current, bool):
            parsed = value.lower() in {"1", "true", "yes", "on"}
        elif isinstance(current, int) and not isinstance(current, bool):
            parsed = int(value)
        elif isinstance(current, float):
            parsed = float(value)
        elif isinstance(current, Path):
            parsed = Path(value)
        else:
            parsed = value
        cur[leaf] = parsed
    if node is not None:
        return Config.from_dict(node)
    return cfg


def _apply_cli_overrides(cfg: Config, overrides: list[str] | None) -> Config:
    """Apply 'a.b.c=value' overrides from the CLI.

    Values are typed by the *default* field type (model_dump serializes Paths
    to str, so overrides of Path fields must be routed through the model's
    own schema, not the dumped dict's runtime types).
    """
    if not overrides:
        return cfg
    node = cfg.to_dict()
    defaults = Config().to_dict()
    for ov in overrides:
        key, _, value = ov.partition("=")
        cur = node
        parts = key.split(".")
        try:
            for part in parts[:-1]:
                cur = cur[part]
            leaf = parts[-1]
            current = cur[leaf]
        except (KeyError, TypeError):
            raise ValueError(f"Unknown config key in override: '{ov}'") from None
        default_val = defaults
        for part in parts:
            default_val = default_val.get(part) if isinstance(default_val, dict) else None
        null_like = value.strip().lower() == "null"
        if isinstance(default_val, Path):
            parsed = None if null_like else Path(value)
        elif isinstance(current, bool) or isinstance(default_val, bool):
            parsed = value.lower() in {"1", "true", "yes", "on"}
        elif isinstance(current, int) and not isinstance(current, bool):
            parsed = None if null_like else int(value)
        elif isinstance(current, float):
            parsed = None if null_like else float(value)
        elif isinstance(default_val, int) and not isinstance(default_val, bool):
            parsed = None if null_like else int(value)
        elif isinstance(default_val, float):
            parsed = None if null_like else float(value)
        else:
            parsed = None if null_like else value
        cur[leaf] = parsed
    return Config.from_dict(node)


def load_config(
    path: str | Path | None = None, overrides: list[str] | None = None
) -> Config:
    """Load a Config from a YAML file with env + CLI overrides applied."""
    cfg = Config()  # defaults
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        cfg = Config.from_dict(data)
    cfg = _apply_env(cfg)
    cfg = _apply_cli_overrides(cfg, overrides)
    return cfg