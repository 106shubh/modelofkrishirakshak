"""Semi-supervised retraining contracts (§3).

The abstention design produces three data qualities for free; this module
encodes their handling as explicit, named contracts so the retraining loop
can't silently misuse them:

  - officer-resolved escalations  → gold labels (use first)
  - auto-resolved detections      → pseudo-labels ONLY above PSEUDO_LABEL_THRESHOLD,
    which is deliberately named and stricter than the escalation threshold —
    reusing the escalation threshold here would let the model confirm its own
    possibly-wrong patterns (feedback loop). A named constant makes that bar
    visible in review instead of an implicit side effect.
  - farmer feedback               → noisy weak labels

A human signs off before any retrained model is promoted; the gate here is a
hard, fail-closed regression check on FIELD-condition accuracy, not accuracy
in general.
"""
from __future__ import annotations

from dataclasses import dataclass

# §3: pseudo-labels are harvested ONLY above this bar. Deliberately stricter
# than EvalConfig.confidence_threshold (0.65) — see module docstring.
PSEUDO_LABEL_THRESHOLD = 0.85


@dataclass
class TrainingExample:
    features: object  # backend row / image ref; opaque here
    label: str | None
    quality: str  # "gold" | "pseudo" | "weak"
    confidence: float | None = None


def pseudo_label_filter(examples: list[TrainingExample]) -> list[TrainingExample]:
    """Keep auto-resolved examples above the named pseudo-label bar."""
    return [
        e
        for e in examples
        if e.quality == "pseudo"
        and e.confidence is not None
        and e.confidence >= PSEUDO_LABEL_THRESHOLD
    ]


def promotion_gate(
    candidate_field_acc: float,
    production_field_acc: float,
    approved_by: str | None,
) -> bool:
    """§3 hard gate: candidate must not regress field-condition accuracy vs the
    production model, AND a human must have signed off. Fail-closed: missing
    approval or missing metrics block promotion.

    Note the metric: FIELD-condition accuracy. Lab-condition wins don't count —
    the abstention mechanism depends on calibration holding where the system
    actually operates.
    """
    if not approved_by or not str(approved_by).strip():
        raise ValueError("promotion requires human sign-off (approved_by)")
    for name, val in (("candidate_field_acc", candidate_field_acc), ("production_field_acc", production_field_acc)):
        if val is None:
            raise ValueError(f"promotion gate requires {name} — fail-closed, not a soft check")
    return candidate_field_acc >= production_field_acc


# §4 reward-history contract lives with the bandits; §3's cadence (monthly
# retrain job) is backend scheduling, not ML logic — recorded here as the
# documented contract: retrain monthly, mix gold + filtered pseudo + weak.
RETRAIN_CADENCE_DAYS = 30
