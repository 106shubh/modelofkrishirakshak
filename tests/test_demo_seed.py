"""End-to-end test for the demo seed case (scripts/seed_demo.py).

Runs the script OFFLINE (--skip-live: the live legs need network; --stub: the
real-model path needs the local dataset+checkpoint) and asserts the escalation
story in the produced artifact. The real-model path (default) is exercised by
running the script manually before the demo.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "seed_demo.py"


def test_seed_demo_escalates_on_weather_alone(tmp_path: Path):
    out = tmp_path / "seed_case.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out), "--skip-live", "--stub"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=300,
    )
    assert proc.returncode == 0, f"seed_demo failed:\n{proc.stdout}\n{proc.stderr}"

    assert out.exists()
    artifact = json.loads(out.read_text(encoding="utf-8"))
    assert artifact["model_stubbed"] is True
    assert artifact["live_scenarios"] == []  # offline run: live legs skipped

    scenarios = {s["scenario"]: s for s in artifact["scenarios"]}

    # The vision model sees a healthy leaf in BOTH runs.
    assert scenarios["monsoon"]["model_class"] == "Tomato___healthy"
    assert scenarios["dry_spell"]["model_class"] == "Tomato___healthy"

    # Monsoon: cool + wet conditions alone ⇒ rule layer high ⇒ escalate.
    mono = scenarios["monsoon"]["causal"]
    assert mono["risk_tier"] == "high"
    assert mono["flagged"] is True
    assert scenarios["monsoon"]["abstain"] is True
    assert any("Rule layer advisory" in a for a in scenarios["monsoon"]["advisories"])

    # Dry spell: same leaf, no risk flags ⇒ auto-resolve. A cold-start advisory
    # is allowed; a rule-layer advisory is not.
    dry = scenarios["dry_spell"]["causal"]
    assert dry["risk_tier"] == "low"
    assert dry["flagged"] is False
    assert scenarios["dry_spell"]["abstain"] is False
    assert all(
        "Rule layer advisory" not in a for a in scenarios["dry_spell"]["advisories"]
    )

    # Issue 5 contract visible in both: zero history ⇒ cold_start flag, not silence.
    assert scenarios["monsoon"]["causal"]["cold_start"] is True
    assert scenarios["dry_spell"]["causal"]["cold_start"] is True

    # Provenance recorded on every scenario.
    assert all("weather_source" in s for s in artifact["scenarios"])
