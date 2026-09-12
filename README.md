# CropGuard — Crop Disease & Pest Diagnosis (SIH26131)

Causal, abstention-aware crop disease and pest diagnosis service. A vision
model's read is never the whole answer: an explicit rule layer encodes plant
pathology thresholds (humidity, rainfall, temperature windows) and can flag or
escalate risk **even when the model is confident**, and a novelty layer catches
images that match nothing the model knows.

- **Dual-model router** — disease classifier + YOLOv8 pest head behind one
  interface (NFR4 adapter contract); hint runs one model, no hint runs both and
  picks by confidence.
- **Calibrated confidence** — temperature scaling fitted at train time, stamped
  into the checkpoint, loaded at serving.
- **Novelty detection** — k-NN distance over the embedding space; a hit becomes
  `escalation_reason: novel_presentation`, distinct from `low_confidence`.
- **Causal rule layer (FR5)** — hardcoded agronomy thresholds, independent of
  the model. High-risk flags always attach an advisory to the farmer response
  and can force escalation (Issue 4 precedence).
- **Live weather (FR3/NFR2)** — Open-Meteo fetch (no key needed), cached per
  taluka + 3h bucket in Redis when `REDIS_URL` is set; falls back to seasonal
  synthesis offline. `weather_source` in every response records provenance.
- **Cold-start honesty (Issue 5)** — zero regional history is surfaced as a
  `cold_start` flag + advisory, never silently read as "no risk".

## Repository layout

```
configs/            default.yaml (defaults), demo.yaml (SIH demo), cpu_smoke.yaml
scripts/            download_plantvillage.py, seed_demo.py (demo seed case)
src/cropguard/
  taxonomy.py       38-class label space, crop ids, backend pest_disease_id mapping
  config.py         pydantic config; YAML + env + CLI overrides
  data/             deterministic PlantVillage dataset, splits, transforms
  models/           backbones (efficientnet/convnext/vit/swin), multimodal fusion, pest_detector
  training/         train.py, evaluate.py (dual-split report), predict.py
  inference/        service.py (FastAPI), engine.py (router/severity), rules.py,
                    novelty.py, weather.py, stubs.py, rule_validation.py
tests/              offline pytest suite (144 tests)
```

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash; .venv/bin/activate on POSIX
pip install -e ".[dev]"
```

Download PlantVillage (lab-condition dataset, ~54k images, 38 classes):

```bash
python scripts/download_plantvillage.py   # → data/raw/PlantVillage/raw/color/
```

Optional: set `REDIS_URL` for the shared weather cache. Without it the service
degrades to per-process caching — never crashes.

## Training

All hyperparameters live in `configs/default.yaml`; override per run with
`--set section.key=value`.

Smoke run (minutes, CPU):

```bash
python -m cropguard.training.train --config configs/cpu_smoke.yaml
```

The run used for the current demo checkpoint (efficientnet_v2_s, label
smoothing, novelty fitting, ~75 min on 8 CPU cores):

```bash
python -m cropguard.training.train --config configs/default.yaml \
  --output-dir checkpoints/v2s_real \
  --set model.backbone=efficientnet_v2_s eval.label_smoothing=0.1 \
       eval.novelty.enabled=true data.train_subset_per_class=150 \
       data.val_subset_per_class=50 training.epochs=3
```

Training always writes `best_model.pth`, fits the calibration temperature on
the configured split (`eval.calibration_split`), and — with novelty enabled —
stamps `novelty_threshold` + reference embeddings into every checkpoint.

Evaluation reports lab-condition and field-condition splits **separately**
(never a blended number) as soon as `data.field_root` points at a field
dataset (PlantDoc-style). Per NFR5, the fitted temperature is only as good as
its split — refit on field data when it exists.

## Serving

```bash
# Defaults (auto-discovers checkpoints/best_model.pth, synthesis weather)
python -m cropguard.inference.service

# SIH demo: real trained checkpoint + live Open-Meteo weather
python -m cropguard.inference.service --config configs/demo.yaml --host 127.0.0.1 --port 8100
```

Endpoints: `GET /health`, `GET /model-info`, `GET /config`,
`POST /predict` (JSON, base64 `image_bytes`), `POST /predict/file` (multipart,
the farmer-app route). Responses carry `class_name`, calibrated `confidence`,
`severity`/`severity_score`, `abstain` + `escalation_reason`
(`novel_presentation` > `rule_high_risk` > `low_confidence`), farmer-facing
`advisories`, and `context` with the weather used + `weather_source`.

## Demo-day runbook

1. **Seed the story** (writes `data/demo/seed_case.json`, gates its own
   invariants, needs network for the live legs):

   ```bash
   python scripts/seed_demo.py            # add --skip-live offline
   ```

   Produces two kinds of proof: **LIVE** — two real districts fetched from
   Open-Meteo in parallel (wet Kolhapur flags medium fungal risk on a
   confident-healthy read; dry Solapur auto-resolves) — and **CONTROLLED** —
   the same leaf under monsoon vs dry-spell payloads: monsoon escalates
   (`abstain=true`, high risk), dry spell resolves, identical model output both
   times. The controlled pair carries the demo on any day, in any season,
   offline included.

2. **Boot the service**: `python -m cropguard.inference.service --config configs/demo.yaml`

3. **The five beats** (verified end-to-end on the live server):
   - `/health` → `status: ok`, `/model-info` → 38 classes, v2s_real checkpoint
   - **Headline**: healthy tomato leaf @ Kolhapur, no weather field → live
     Open-Meteo sky drives the rule layer; confident `Tomato___healthy` (~0.99)
     still gets a medium fungal-risk advisory attached (Issue 4)
   - **Contrast**: byte-identical leaf @ Solapur → low risk, auto-resolve.
     *Same pixels — the rule layer decides, not the model.*
   - **Diagnosis**: late-blight leaf + monsoon payload → `Tomato___Late_blight`
     (~0.99), severity high, high-risk advisory with contributing factors
   - **Farmer route**: multipart upload to `/predict/file` → same verdict

4. **Demo-day notes** (from the live rehearsal):
   - First live-weather call ≈ 1.7–2.0 s (Open-Meteo RTT); model-only
     inference ≈ 0.08 s. If the venue network dies, live legs fall back to
     `synthesize` (provenance shows it) and the controlled pair carries the show.
   - Windows consoles: run scripts with `PYTHONIOENCODING=utf-8` (the server
     itself is unaffected).
   - The seed script runs the **real model on a real `Tomato___healthy` dataset
     photo** by default; `--stub` forces the offline synthetic-leaf mode.

## Tests

```bash
python -m pytest tests/        # 144 tests, fully offline by construction
```

Hermeticity is enforced, not hoped for: the test fixture pins
`weather.provider=synthesize`, the demo test runs `--skip-live --stub`, models
build with `pretrained=False`, and the redis-failure test targets a loopback
port. CI (`.github/workflows/ci.yml`) runs the same suite on every push with
CPU-only torch wheels.

## Honest caveats

- The 98.2% val accuracy and the fitted temperature (T=0.632) are
  **lab-condition** (PlantVillage) numbers. Field-condition validation needs a
  field split (`data.field_root`) — the dual-split evaluation is ready for it.
- The pest head ships with a documented dark-spot heuristic stand-in
  (`model.pest_detector: heuristic`); `yolo` activates the moment trained
  weights exist (`inference.pest_weights`).
- Outbreak clustering, IPM selection, jurisdiction scoping, and the audit log
  are `stubs.py` contracts awaiting the backend tables.
- Weather synthesis is a seasonal climatology prior, not a forecast — treat
  `weather_source: synthesize` responses as demo-grade only.
