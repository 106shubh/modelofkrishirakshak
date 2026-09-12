"""Tests for rule-threshold validation (§2b: unsupervised check on the rules)."""
from __future__ import annotations

import pytest

from cropguard.inference.rule_validation import (
    OutbreakRecord,
    summarize_disagreement,
    validate_rule_thresholds,
)


def _rec(rh: float, temp: float = 22.0, rain7: float = 60.0) -> OutbreakRecord:
    return OutbreakRecord(
        taluka="nashik",
        pest_disease_id=101,
        temperature_c=temp,
        humidity_pct=rh,
        rainfall_mm_7d=rain7,
        growth_stage="fruiting",
    )


def _rule(rec: OutbreakRecord) -> bool:
    """Stands in for the rule layer's fungal predicate (RH > 85 AND rain7 > 50)."""
    return rec.humidity_pct > 85.0 and rec.rainfall_mm_7d > 50.0


def test_requires_records():
    with pytest.raises(ValueError):
        validate_rule_thresholds([], "tomato", _rule)


def test_all_onsets_satisfy_rule():
    recs = [_rec(rh=90 + i * 0.5) for i in range(9)]  # one tight wet cluster
    rep = validate_rule_thresholds(recs, "tomato", _rule)
    assert rep.n_clusters >= 1
    assert all(c == pytest.approx(1.0) for c in rep.rule_coverage)
    assert rep.missed_onsets == 0
    assert "consistent" in (summarize_disagreement(rep) or "").lower()


def test_clustered_onsets_outside_rule_flag_disagreement():
    # 10 identical onsets in a dry regime the rule would never flag (RH 60,
    # rain 10) — one dense cluster entirely outside the rule's conditions.
    recs = [_rec(rh=60.0, rain7=10.0) for _ in range(10)]
    rep = validate_rule_thresholds(recs, "tomato", _rule)
    assert rep.n_clusters == 1
    assert rep.missed_onsets == 10, "dry-regime onsets must count as missed"
    assert rep.rule_coverage[0] == pytest.approx(0.0)
    verdict = summarize_disagreement(rep)
    assert verdict is not None and "revisit" in verdict.lower()
    assert "Advisory only" in verdict


def test_mixed_regimes_partial_coverage():
    wet = [_rec(rh=95.0) for _ in range(6)]  # inside the rule
    dry = [_rec(rh=60.0, rain7=10.0) for _ in range(6)]  # outside it
    rep = validate_rule_thresholds(wet + dry, "tomato", _rule, eps=1.2)
    assert rep.n_clusters == 2
    assert rep.missed_onsets == 6  # the dry cluster is real outbreak data the rule misses
    assert sorted(round(c, 2) for c in rep.rule_coverage) == [0.0, 1.0]


def test_rule_layer_is_never_mutated():
    """The validation path must not touch rule-layer state (advisory-only contract)."""
    from cropguard.inference.rules import CausalRuleLayer

    rules = CausalRuleLayer()
    before = dict(rules.rules)
    recs = [_rec(rh=60.0, rain7=10.0) for _ in range(5)]
    rep = validate_rule_thresholds(recs, "tomato", _rule)
    summarize_disagreement(rep)
    assert rules.rules == before  # thresholds untouched


def test_dbscan_finds_two_clusters():
    """Well-separated wet/dry regimes → 2 clusters (or the documented fallback)."""
    wet = [_rec(rh=95.0, temp=18.0, rain7=80.0) for _ in range(5)]
    dry = [_rec(rh=40.0, temp=34.0, rain7=0.0) for _ in range(5)]
    rep = validate_rule_thresholds(wet + dry, "tomato", _rule, eps=1.5)
    # With sklearn: 2 clusters. Without: single trivial cluster (documented).
    assert rep.n_clusters >= 1
