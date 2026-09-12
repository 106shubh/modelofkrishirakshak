"""Causal rule layer: encoded plant pathology knowledge independent of ML."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..data.metadata import WeatherVector
from ..taxonomy import TAXONOMY, DiseaseInfo

log = logging.getLogger(__name__)


@dataclass
class RuleResult:
    """Outcome of the causal rule layer."""
    risk_tier: str  # low | medium | high
    flagged: bool
    reason: str
    contributing_factors: list[str]


class CausalRuleLayer:
    """Implements hardcoded domain thresholds for disease/pest risk."""

    def __init__(self) -> None:
        # Thresholds based on standard agronomy (e.g. Late Blight: cool + wet)
        self.rules = {
            "fungal_general": self._rule_fungal_general,
            "late_blight": self._rule_late_blight,
            "spider_mites": self._rule_spider_mites,
        }

    def evaluate(
        self,
        crop_id: int,
        weather: WeatherVector,
        growth_stage: str,
        predicted_class: str | None = None,
    ) -> RuleResult:
        """Evaluate risk based on weather and context."""
        factors = []
        highest_tier = "low"
        reasons = []

        # 1. Run general fungal rule
        res_fungal = self._rule_fungal_general(weather)
        if res_fungal.flagged:
            factors.extend(res_fungal.contributing_factors)
            reasons.append(res_fungal.reason)
            highest_tier = self._max_tier(highest_tier, res_fungal.risk_tier)

        # 2. Run specific rules if predicted_class matches. Detector/unknown
        # class names (outside the taxonomy) skip disease-specific rules but
        # still get the general weather rules.
        if predicted_class:
            try:
                info = TAXONOMY.info(predicted_class)
            except KeyError:
                info = None
            if "late_blight" in predicted_class.lower():
                res_lb = self._rule_late_blight(weather)
                if res_lb.flagged:
                    factors.extend(res_lb.contributing_factors)
                    reasons.append(res_lb.reason)
                    highest_tier = self._max_tier(highest_tier, res_lb.risk_tier)
            
            if info is not None and info.disease_type == "pest" and "mite" in predicted_class.lower():
                res_mite = self._rule_spider_mites(weather)
                if res_mite.flagged:
                    factors.extend(res_mite.contributing_factors)
                    reasons.append(res_mite.reason)
                    highest_tier = self._max_tier(highest_tier, res_mite.risk_tier)

        # 3. Growth stage factor (e.g. fruiting stage is high risk for many diseases)
        if growth_stage.lower() in {"flowering", "fruiting"}:
            factors.append(f"Vulnerable growth stage: {growth_stage}")
            if highest_tier == "low":
                highest_tier = "medium"

        return RuleResult(
            risk_tier=highest_tier,
            flagged=highest_tier != "low",
            reason="; ".join(reasons) if reasons else "Normal conditions",
            contributing_factors=list(set(factors)),
        )

    def _max_tier(self, t1: str, t2: str) -> str:
        order = {"low": 0, "medium": 1, "high": 2}
        return t1 if order[t1] >= order[t2] else t2

    def _rule_fungal_general(self, w: WeatherVector) -> RuleResult:
        factors = []
        if w.humidity_pct > 85.0:
            factors.append("High humidity (>85%)")
        if w.rainfall_mm_7d > 50.0:
            factors.append("Sustained 7d rainfall (>50mm)")
        
        if len(factors) >= 2:
            return RuleResult("high", True, "High fungal pressure", factors)
        if len(factors) == 1:
            return RuleResult("medium", True, "Elevated fungal risk", factors)
        return RuleResult("low", False, "", [])

    def _rule_late_blight(self, w: WeatherVector) -> RuleResult:
        # Late Blight (Phytophthora infestans) prefers cool, very wet weather.
        # Smith Periods: 2+ days with min temp >10C and 11+ hours RH >90%
        factors = []
        is_cool = 10.0 <= w.temperature_c <= 24.0
        is_wet = w.humidity_pct > 90.0 or w.rainfall_mm_24h > 10.0

        if is_cool and is_wet:
            factors.append("Ideal Late Blight window (Cool + Wet)")
            return RuleResult("high", True, "Late Blight epidemic risk", factors)
        return RuleResult("low", False, "", [])

    def _rule_spider_mites(self, w: WeatherVector) -> RuleResult:
        # Mites thrive in hot, dry weather
        factors = []
        if w.temperature_c > 32.0 and w.humidity_pct < 50.0:
            factors.append("Hot, dry conditions (favors mite explosion)")
            return RuleResult("high", True, "Pest outbreak risk (Mites)", factors)
        return RuleResult("low", False, "", [])
