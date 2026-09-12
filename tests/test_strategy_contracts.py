"""Tests for §3 semi-supervised contracts and §4 bandit skeletons."""
from __future__ import annotations

import pytest

from cropguard.bandits import AbstentionBandit, IPMRankingBandit
from cropguard.inference.stubs import IPMRecommendation
from cropguard.semi import PSEUDO_LABEL_THRESHOLD, TrainingExample, promotion_gate, pseudo_label_filter


# --------------------------------------------------------------------------- #
# §3: semi-supervised contracts
# --------------------------------------------------------------------------- #


def test_pseudo_label_threshold_is_stricter_than_escalation():
    from cropguard.config import Config

    assert PSEUDO_LABEL_THRESHOLD > Config().eval.confidence_threshold, (
        "§3: the pseudo-label bar MUST be named and stricter than escalation"
    )


def test_pseudo_label_filter():
    examples = [
        TrainingExample("a", "blight", "pseudo", confidence=0.90),  # above bar
        TrainingExample("b", "blight", "pseudo", confidence=0.70),  # below bar
        TrainingExample("c", "rust", "gold", confidence=0.55),      # gold: always kept out of this filter
        TrainingExample("d", "rust", "pseudo", confidence=None),    # no confidence → excluded
    ]
    kept = pseudo_label_filter(examples)
    assert [e.features for e in kept] == ["a"]


def test_promotion_gate_requires_signoff():
    with pytest.raises(ValueError, match="sign-off"):
        promotion_gate(0.80, 0.75, approved_by=None)


def test_promotion_gate_requires_metrics_fail_closed():
    with pytest.raises(ValueError, match="candidate_field_acc"):
        promotion_gate(None, 0.75, approved_by="officer-1")
    with pytest.raises(ValueError, match="production_field_acc"):
        promotion_gate(0.80, None, approved_by="officer-1")


def test_promotion_gate_blocks_regression():
    assert promotion_gate(0.82, 0.80, approved_by="officer-1") is True
    with pytest.raises(AssertionError):
        assert promotion_gate(0.79, 0.80, approved_by="officer-1") is True


# --------------------------------------------------------------------------- #
# §4: IPM ranking bandit — the wall
# --------------------------------------------------------------------------- #


def _catalog():
    # Expert prior order: bio-strong ranks first (10 < 20).
    return [
        IPMRecommendation("bio-strong", 101, "high", 10, "biological"),
        IPMRecommendation("bio-weak", 101, "high", 20, "biological"),
        IPMRecommendation("chem-a", 101, "high", 30, "chemical"),
    ]


def test_cold_bandit_preserves_expert_order():
    result = IPMRankingBandit().refine_ranking((101, "high"), _catalog())
    assert [r.recommendation_id for r in result] == ["bio-strong", "bio-weak", "chem-a"]


def test_bandit_reorders_within_tier_only():
    b = IPMRankingBandit()
    # bio-weak confirmed effective 8/10; bio-strong fails 6/10 → within
    # "biological" the bandit may swap them…
    for _ in range(8):
        b.record_outcome("bio-weak", True, {"crop": 1})
    for _ in range(6):
        b.record_outcome("bio-strong", False, {"crop": 1})
    result = b.refine_ranking((101, "high"), _catalog())
    ids = [r.recommendation_id for r in result]
    assert ids[:2] == ["bio-weak", "bio-strong"]  # within-tier reorder happened
    # …but chemistry can never cross the wall regardless of reward:
    assert ids[-1] == "chem-a"
    assert b.changes and b.changes[-1].reason == "thompson-within-tier"


def test_chemical_reward_can_never_promote_it_above_biological():
    b = IPMRankingBandit()
    for _ in range(50):
        b.record_outcome("chem-a", True, {"crop": 1})       # chemical always works
        b.record_outcome("bio-weak", False, {"crop": 1})    # biological always fails
    result = b.refine_ranking((101, "high"), _catalog())
    ids = [r.recommendation_id for r in result]
    assert ids.index("chem-a") > ids.index("bio-weak"), "wall violated: chemical above biological"
    assert ids.index("chem-a") > ids.index("bio-strong")


def test_enforce_wall_restores_ordering_after_any_source():
    shuffled = [
        IPMRecommendation("chem-a", 101, "high", 1, "chemical"),
        IPMRecommendation("bio-strong", 101, "high", 2, "biological"),
    ]
    fixed = IPMRankingBandit.enforce_wall(shuffled)
    assert [r.recommendation_id for r in fixed] == ["bio-strong", "chem-a"]


def test_ranking_changes_are_logged():
    b = IPMRankingBandit()
    b.record_outcome("bio-weak", True, {})
    b.refine_ranking((101, "high"), _catalog())
    assert len(b.reward_history) == 1  # reward history that produced any change


# --------------------------------------------------------------------------- #
# §4: abstention threshold bandit
# --------------------------------------------------------------------------- #


def test_abstention_bandit_cold_start_returns_base():
    b = AbstentionBandit()
    assert b.propose_threshold(0.65, {"crop": 1}) == 0.65


def test_abstention_bandit_adapts_and_stays_bounded():
    b = AbstentionBandit()
    # Escalations keeps getting overturned (low reward) → demand more confidence.
    for _ in range(20):
        b.record_outcome({"crop": 1, "pest": 101}, "escalate", reward=0.1)
    thr = b.propose_threshold(0.65, {"crop": 1, "pest": 101})
    assert thr > 0.65
    assert 0.5 <= thr <= 0.95  # never proposes an unsafe value
    assert len(b.history) == 20  # reward history logged for every outcome
