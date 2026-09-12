"""Demo seed case: weather flags risk before symptoms are visible (FR5/FR6).

Two kinds of proof, one artifact:

1. LIVE NOW — two real Maharashtra districts with contrastingly wet/dry
   climates are submitted with NO weather field: the service fetches actual
   conditions from Open-Meteo. Whatever today's sky does, the rule layer's
   verdict is traceable to real, timestamped data. (Wet=dense: Kolhapur;
   dry=leeward: Solapur.)

2. CONTROLLED PAIR — the SAME healthy leaf under monsoon vs dry-spell weather.
   Identical model output both times; only the weather differs. Monsoon
   escalates to an officer (cool + wet = Late Blight window, high fungal
   pressure) while the vision model still reads "healthy"; the dry spell
   auto-resolves. This pair is what makes the escalation provably the rule
   layer's decision, not the model's — it holds on any day, in any season.

The demo runs the real trained model by default (checkpoints/v2s_real —
efficientnet_v2_s, calibrated, novelty-fitted) on a real Tomato___healthy leaf
from the dataset, so the premise is a genuine confident-healthy read. With
--stub (or no checkpoint/dataset), the vision model is stubbed to a confident
"healthy" read on a synthetic green leaf — same flow, fully offline.

Usage:
    python scripts/seed_demo.py                       # needs network for leg 1
    python scripts/seed_demo.py --skip-live           # offline: leg 2 only
    python scripts/seed_demo.py --stub                # force the stub model
    python scripts/seed_demo.py --out other.json
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Make `src` importable when run as a script (repo layout: src/cropguard/...).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from cropguard.config import Config, load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("seed_demo")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Real-model default (SIH demo): the trained v2s_real checkpoint. Falling back
# to auto-discovery/stub keeps the script runnable on a fresh clone.
DEFAULT_CKPT = PROJECT_ROOT / "checkpoints" / "v2s_real" / "best_model.pth"
HEALTHY_LEAF_DIR = PROJECT_ROOT / "data" / "raw" / "PlantVillage" / "raw" / "color" / "Tomato___healthy"

# Live legs: contrasting districts, no explicit weather → fetched for real.
LIVE_LEGIONS = ["kolhapur", "solapur"]  # wettest vs driest in REGIONS

# Controlled regimes (must satisfy the rules.py thresholds listed in brackets).
REGIMES = {
    "monsoon": {  # Late Blight window: 10-24C & wet; fungal: RH>85% + rain7d>50mm
        "temperature_c": 18.0,
        "humidity_pct": 95.0,
        "rainfall_mm_24h": 12.0,
        "rainfall_mm_7d": 85.0,
    },
    "dry_spell": {  # satisfies neither rule
        "temperature_c": 33.0,
        "humidity_pct": 40.0,
        "rainfall_mm_24h": 0.0,
        "rainfall_mm_7d": 0.0,
    },
}

# Escalation invariants for the CONTROLLED pair (mirrored in test_demo_seed).
EXPECTED = {
    "monsoon": {"abstain": True, "risk_tier": "high", "cold_start": True},
    "dry_spell": {"abstain": False, "risk_tier": "low", "cold_start": True},
}


def _healthy_leaf_jpeg(real: bool) -> bytes:
    """Demo leaf bytes. Real model → a real Tomato___healthy dataset photo (the
    premise must be a GENUINE confident-healthy read); stub → a deterministic
    synthetic green leaf."""
    if real and HEALTHY_LEAF_DIR.is_dir():
        leaf = sorted(HEALTHY_LEAF_DIR.glob("*.jpg"))[0]
        log.info("Demo leaf: real dataset image %s", leaf.name)
        return leaf.read_bytes()
    img = Image.new("RGB", (256, 256), color=(90, 140, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _stub_model_unless_available(cfg: Config, force_stub: bool):
    """Use the real checkpoint (default: v2s_real) unless --stub or none exists.

    The stub returns a confident, healthy prediction regardless of image —
    same output contract as MultimodalClassifier.forward (logits/probs).
    """
    try:
        if force_stub:
            raise FileNotFoundError("--stub requested")
        ckpt = cfg.inference.checkpoint or (cfg.training.out_dir / "best_model.pth")
        if not Path(ckpt).exists():
            raise FileNotFoundError(ckpt)
        return None  # sentinel: app lifespan loads the real checkpoint
    except FileNotFoundError:
        log.info("Stubbing vision model (demo premise: healthy leaf, nothing to see)")

        import torch

        from cropguard.taxonomy import TAXONOMY

        n = len(TAXONOMY.classes)
        healthy_idx = TAXONOMY.classes.index("Tomato___healthy")

        class _StubRouter:
            def predict(self, image, type_hint=None, **context):
                logits = torch.zeros(n)
                logits[healthy_idx] = 6.0  # confident healthy read
                probs = torch.softmax(logits, dim=-1)
                return {
                    "logits": logits.unsqueeze(0),
                    "probs": probs.unsqueeze(0),
                    "features": torch.zeros(1, 1),
                }

        return _StubRouter()


def _post_predict(client: TestClient, leaf_b64: str, payload: dict) -> tuple[dict, float]:
    t0 = time.perf_counter()
    r = client.post("/predict", json={**payload, "image_bytes": leaf_b64})
    r.raise_for_status()
    return r.json(), time.perf_counter() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "demo" / "seed_case.json")
    parser.add_argument("--skip-live", action="store_true", help="skip the live-weather legs (offline)")
    parser.add_argument("--stub", action="store_true", help="force the stub vision model (offline demo premise)")
    args = parser.parse_args()

    from cropguard.inference.service import create_app

    cfg = load_config(args.config)
    if DEFAULT_CKPT.exists():
        cfg.inference.checkpoint = DEFAULT_CKPT  # SIH demo default: real model
    cfg.weather.provider = "open-meteo"  # live legs need the real fetch

    app = create_app(cfg)
    stub = _stub_model_unless_available(cfg, force_stub=args.stub)
    leaf_b64 = base64.b64encode(_healthy_leaf_jpeg(real=stub is None)).decode("ascii")

    live_scenarios: list[dict] = []
    scenarios: list[dict] = []

    with TestClient(app) as client:
        if stub is not None:
            # Lifespan attempted the real load and failed (no checkpoint);
            # override with the stub for the demo premise.
            app.state.router = stub
            app.state.ready = True

        base_payload = {
            "crop_id": 1,  # tomato
            "stage": "vegetative",
            "type_hint": "disease",
        }

        # ---- Leg 1: LIVE weather, two contrasting districts, in parallel ---- #
        if not args.skip_live:
            def run_live(region: str) -> tuple[str, dict, float]:
                body, dt = _post_predict(
                    client, leaf_b64, {**base_payload, "region": region}  # no "weather" key
                )
                return region, body, dt

            with ThreadPoolExecutor(max_workers=len(LIVE_LEGIONS)) as pool:
                results = list(pool.map(run_live, LIVE_LEGIONS))
            # Longest call first = the headline: real weather → real verdict.
            results.sort(key=lambda x: x[2], reverse=True)
            for region, body, dt in results:
                live_scenarios.append(
                    {
                        "scenario": f"live_{region}",
                        "region": region,
                        "latency_sec": round(dt, 3),
                        "weather": body["context"]["weather"],
                        "weather_source": body["context"]["weather_source"],
                        "model_class": body["class_name"],
                        "model_confidence": body["confidence"],
                        "severity": body["severity"],
                        "abstain": body["abstain"],
                        "advisories": body["advisories"],
                        "causal": body["causal_analysis"],
                    }
                )
                log.info(
                    "live/%s: %.2fs src=%s weather=%s → model=%s conf=%.2f risk=%s abstain=%s",
                    region, dt, body["context"]["weather_source"], body["context"]["weather"],
                    body["class_name"], body["confidence"],
                    body["causal_analysis"]["risk_tier"], body["abstain"],
                )

        # ---- Leg 2: CONTROLLED pair — guaranteed escalate-vs-resolve contrast ---- #
        for name, weather in REGIMES.items():
            body, _ = _post_predict(
                client, leaf_b64, {**base_payload, "region": "nashik", "weather": weather}
            )
            scenarios.append(
                {
                    "scenario": name,
                    "weather": weather,
                    "weather_source": body["context"]["weather_source"],
                    "model_class": body["class_name"],
                    "model_confidence": body["confidence"],
                    "severity": body["severity"],
                    "abstain": body["abstain"],
                    "advisories": body["advisories"],
                    "causal": body["causal_analysis"],
                }
            )
            log.info(
                "%s: model=%s conf=%.2f abstain=%s risk=%s",
                name, body["class_name"], body["confidence"], body["abstain"],
                body["causal_analysis"]["risk_tier"],
            )

    # ---- Invariants (script-level gate; mirrored in test_demo_seed) ---- #
    for s in live_scenarios:
        assert s["model_class"] == "Tomato___healthy", "demo premise: healthy leaf"
        assert s["weather_source"] in {"api", "synthesize", "cache"}, "provenance missing"
        assert s["causal"]["cold_start"] is True, "Issue 5: zero history must be flagged"
        if s["weather_source"] == "synthesize":
            log.warning("live leg %s fell back to synthesis (API unreachable?)", s["region"])
        assert s["causal"]["flagged"] == (
            s["causal"]["risk_tier"] in {"medium", "high"}
        ), "rule layer verdict must match live conditions"

    by_name = {s["scenario"]: s for s in scenarios}
    for name, expected in EXPECTED.items():
        s = by_name[name]
        assert s["abstain"] == expected["abstain"], f"{name}: abstain={s['abstain']}"
        assert s["causal"]["risk_tier"] == expected["risk_tier"], f"{name}: risk={s['causal']['risk_tier']}"
        assert s["causal"]["cold_start"] == expected["cold_start"], f"{name}: cold_start mismatch"
    assert by_name["monsoon"]["causal"]["flagged"] is True
    assert any("Rule layer advisory" in a for a in by_name["monsoon"]["advisories"]), (
        "Issue 4 violated: high-risk rule flag missing from farmer-facing advisories"
    )
    assert by_name["dry_spell"]["causal"]["flagged"] is False
    assert by_name["monsoon"]["model_class"] == by_name["dry_spell"]["model_class"], (
        "controlled pair premise: the leaf read must be identical both times"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "title": "Weather flags risk before symptoms are visible",
        "premise": (
            "A healthy-looking tomato leaf under two lenses. LIVE: real district "
            "weather fetched from Open-Meteo drives the rule layer's verdict on "
            "today's conditions. CONTROLLED: the same leaf under monsoon vs dry-"
            "spell weather — identical model output, only the weather differs; "
            "monsoon escalates, dry spell auto-resolves. Escalation traces to "
            "encoded pathology thresholds, not model output."
        ),
        "model_stubbed": stub is not None,
        "live_scenarios": live_scenarios,
        "scenarios": scenarios,
    }
    args.out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    log.info("Seed artifact written to %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
