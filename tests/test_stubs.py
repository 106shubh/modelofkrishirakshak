"""Tests for backend-seam stubs: clustering, jurisdiction scope, audit, IPM, history."""
from __future__ import annotations

from datetime import datetime, timedelta

from cropguard.inference.stubs import (
    AuditLog,
    Detection,
    IPMRecommendation,
    cluster_detections,
    fetch_regional_history,
    scope_by_jurisdiction,
    select_ipm_recommendations,
)


def _det(pid: int, taluka: str, day: int, did: str | None = None) -> Detection:
    return Detection(
        detection_id=did or f"d{day}-{pid}-{taluka}",
        pest_disease_id=pid,
        taluka=taluka,
        detected_at=datetime(2026, 9, 1) + timedelta(days=day),
    )


# --------------------------------------------------------------------------- #
# FR8: outbreak clustering
# --------------------------------------------------------------------------- #


def test_cluster_requires_three_same_key():
    dets = [_det(101, "pune", 0), _det(101, "pune", 2), _det(101, "pune", 4)]
    clusters = cluster_detections(dets)
    assert len(clusters) == 1
    assert clusters[0].qualifies() is True
    assert clusters[0].size == 3


def test_cluster_two_detections_do_not_qualify():
    clusters = cluster_detections([_det(101, "pune", 0), _det(101, "pune", 1)])
    assert len(clusters) == 1
    assert clusters[0].qualifies() is False


def test_cluster_splits_on_7_day_gap():
    dets = [_det(101, "pune", 0), _det(101, "pune", 8), _det(101, "pune", 10)]
    clusters = cluster_detections(dets)
    assert len(clusters) == 2  # gap of 8 days breaks the chain
    assert not any(c.qualifies() for c in clusters)


def test_cluster_keys_are_independent():
    dets = [
        _det(101, "pune", 0),
        _det(101, "nashik", 1),  # different taluka
        _det(102, "pune", 2),    # different pest
        _det(101, "pune", 3),
        _det(101, "pune", 4),
    ]
    clusters = cluster_detections(dets)
    target = [c for c in clusters if c.pest_disease_id == 101 and c.taluka == "pune"]
    assert len(target) == 1 and target[0].size == 3 and target[0].qualifies()


def test_cluster_exact_7_day_gap_still_chains():
    clusters = cluster_detections(
        [_det(101, "pune", 0), _det(101, "pune", 7), _det(101, "pune", 8)]
    )
    assert len(clusters) == 1 and clusters[0].qualifies()


# --------------------------------------------------------------------------- #
# NFR3: jurisdiction scoping
# --------------------------------------------------------------------------- #


def test_scope_filters_to_officer_taluka():
    rows = [_det(101, "pune", 0), _det(101, "nashik", 1), _det(102, "pune", 2)]
    scoped = scope_by_jurisdiction(rows, officer_taluka="pune")
    assert len(scoped) == 2 and all(r.taluka == "pune" for r in scoped)


# --------------------------------------------------------------------------- #
# FR9: audit log
# --------------------------------------------------------------------------- #


def test_audit_log_records_mutations():
    audit = AuditLog()
    audit.record("officer-7", "update_recommendation", "ipm_rec:42", "severity high→medium")
    audit.record("admin-1", "disable_user", "user:9")
    assert len(audit.entries) == 2
    e = audit.entries[0]
    assert e.actor_id == "officer-7" and e.action == "update_recommendation"
    assert e.entity == "ipm_rec:42"


# --------------------------------------------------------------------------- #
# FR7: IPM selection
# --------------------------------------------------------------------------- #


def test_ipm_orders_bio_cultural_before_chemical():
    catalog = [
        IPMRecommendation("r3", 101, "high", 30, "chemical", "spray X"),
        IPMRecommendation("r1", 101, "high", 10, "cultural", "remove affected leaves"),
        IPMRecommendation("r2", 101, "high", 20, "biological", "release predator mites"),
        IPMRecommendation("r4", 101, "low", 10, "cultural", "row spacing"),  # wrong tier
        IPMRecommendation("r5", 102, "high", 10, "cultural", "other pest"),  # wrong id
    ]
    picks = select_ipm_recommendations(catalog, 101, "high")
    assert [p.recommendation_id for p in picks] == ["r1", "r2", "r3"]
    assert picks[-1].method == "chemical"  # biological/cultural first


def test_ipm_no_match_returns_empty():
    assert select_ipm_recommendations([], 999, "high") == []


# --------------------------------------------------------------------------- #
# Issue 5: regional history / cold-start contract
# --------------------------------------------------------------------------- #


def test_history_counts_within_window():
    now = datetime(2026, 9, 10)
    dets = [
        _det(101, "pune", 6),   # 2026-09-07, inside 7d of `now`
        _det(101, "pune", 0),   # 2026-09-01, outside window
        _det(101, "nashik", 6), # wrong taluka
        _det(102, "pune", 6),   # wrong pest
    ]
    assert fetch_regional_history(dets, "pune", 101, now=now) == 1


def test_history_zero_means_unknown_not_safe():
    """Contract: 0 = no evidence available (cold start), never 'no risk'."""
    now = datetime(2026, 9, 10)
    assert fetch_regional_history([], "pune", 101, now=now) == 0
