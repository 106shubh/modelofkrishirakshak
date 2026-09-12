"""Empirical validation of causal-rule thresholds (§2 unsupervised, second use).

Runs DBSCAN over historical (temperature, humidity, rainfall) vectors aligned
to confirmed outbreak onsets, then compares the discovered weather patterns
against the hardcoded thresholds in the rule layer.

Contract (from the strategy doc, verbatim in spirit):
  - Disagreement is a SIGNAL to a human to revisit a threshold.
  - This module NEVER mutates rule-layer behavior; interpretability of the
    causal explanation beats marginal accuracy.
  - ponytail: thresholds are compared as scalar coverage margins (RH, 7d rain)
    because that is what rules.py encodes today; duration-style thresholds
    (e.g. "sustained humidity for N days") join when the rule layer takes
    multi-day weather history — see the service's TODO on weather history.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutbreakRecord:
    """One confirmed outbreak onset with the weather at that time/place.

    Mirrors the backend detections/join table shape the strategy doc assumes;
    until it exists, `discovered_conditions` accepts these rows directly.
    """

    taluka: str
    pest_disease_id: int
    temperature_c: float
    humidity_pct: float
    rainfall_mm_7d: float
    growth_stage: str


@dataclass
class ClusterReport:
    """DBSCAN's discovered patterns for one (crop, disease) family."""

    family: str
    n_clusters: int
    n_noise: int
    # Per-cluster: share of onsets satisfying the rule layer's conditions.
    rule_coverage: list[float]
    # Disagreement: onsets clustered as typical outbreak conditions but NOT
    # flagged by the rule layer (the rule would have missed them).
    missed_onsets: int
    total_onsets: int


def _cluster_with_dbscan(vectors, eps: float, min_samples: int) -> list[int] | None:
    """Fit DBSCAN and return per-point labels (-1 = noise). sklearn lazy import
    + graceful absence (documented single-cluster fallback at the caller)."""
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        log.warning("scikit-learn unavailable — rule validation needs it (it is a core dep)")
        return None
    return [int(l) for l in DBSCAN(eps=eps, min_samples=min_samples).fit(vectors).labels_]


def validate_rule_thresholds(
    records: list[OutbreakRecord],
    family: str,
    rule_conditions,
    eps: float = 1.2,
    min_samples: int = 3,
) -> ClusterReport:
    """Cluster outbreak-weather vectors and score the rule layer against them.

    family: cluster key, e.g. "tomato" or "tomato_late_blight" — records are
        assumed pre-filtered to it by the caller (doc: per disease/pest family).
    rule_conditions: callable(OutbreakRecord) -> bool, the EXACT predicate the
        rule layer uses (pass e.g. CausalRuleLayer._rule_fungal_general wrapped
        for the record). Keeping it a callable means the rule layer stays the
        single source of truth — no threshold duplication to drift.

    Returns a ClusterReport. Disagreement (missed_onsets > 0) is informational:
    show it to the domain expert; do not auto-tune the rule.
    """
    if not records:
        raise ValueError("no outbreak records provided")

    vectors = [
        [r.temperature_c, r.humidity_pct, r.rainfall_mm_7d] for r in records
    ]
    labels = _cluster_with_dbscan(vectors, eps=eps, min_samples=min_samples)
    if labels is None:
        # Without sklearn, the honest fallback is a single trivial cluster:
        # the coverage/disagreement math still runs, discovery doesn't.
        labels = [0] * len(records)

    n_clusters = len(set(labels)) - (-1 in labels)
    n_noise = labels.count(-1)

    covered = sum(1 for r in records if rule_conditions(r))
    rule_coverage = [covered / len(records)] if n_clusters == 0 else [
        sum(1 for i, r in enumerate(records) if labels[i] == c and rule_conditions(r))
        / max(1, sum(1 for l in labels if l == c))
        for c in range(n_clusters)
    ]
    missed_onsets = sum(
        1
        for i, r in enumerate(records)
        if labels[i] != -1 and not rule_conditions(r)
    )

    return ClusterReport(
        family=family,
        n_clusters=n_clusters,
        n_noise=n_noise,
        rule_coverage=rule_coverage,
        missed_onsets=missed_onsets,
        total_onsets=len(records),
    )


def summarize_disagreement(report: ClusterReport, revisit_below: float = 0.6) -> str | None:
    """Human-readable verdict for the expert. Pure advisory — returns text.

    revisit_below: if fewer than this share of clustered onsets satisfy the
    rule, recommend revisiting the threshold. 0.6 is a default, not gospel.
    """
    clustered = report.total_onsets - report.n_noise
    if clustered == 0:
        return None
    mean_coverage = sum(report.rule_coverage) / len(report.rule_coverage)
    if mean_coverage < revisit_below:
        return (
            f"[{report.family}] Rule conditions cover only {mean_coverage:.0%} of "
            f"{clustered} clustered onsets ({report.missed_onsets} missed). "
            "Recommend a domain expert revisit the hardcoded threshold. "
            "(Advisory only — the rule layer is unchanged.)"
        )
    return (
        f"[{report.family}] Rule conditions cover {mean_coverage:.0%} of clustered "
        f"onsets — thresholds consistent with observed outbreak data."
    )
