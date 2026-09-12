"""Backend-facing primitives, stubbed at the ML-service seam.

These encode the *contracts* the backend must honor (FR7/FR8/FR9, NFR3, Issue 5)
so the ML service can be demoed and tested against them. Every stub here is
in-memory and session-scoped — the real implementations live behind the backend
detections/ipm_recommendations/audit_log tables.
"""
from __future__ import annotations

import itertools
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

log = logging.getLogger(__name__)

CLUSTER_MIN_DETECTIONS = 3
CLUSTER_WINDOW_DAYS = 7


@dataclass
class Detection:
    """One inference submission, as the backend detections row would carry it."""

    detection_id: str
    pest_disease_id: int
    taluka: str
    detected_at: datetime


# --------------------------------------------------------------------------- #
# FR8: outbreak clustering
# --------------------------------------------------------------------------- #


@dataclass
class OutbreakCluster:
    cluster_id: str
    pest_disease_id: int
    taluka: str
    detection_ids: list[str] = field(default_factory=list)
    started_at: datetime | None = None
    last_seen_at: datetime | None = None
    size: int = 0

    def qualifies(self) -> bool:
        """FR8 threshold: ≥3 detections of the same pest/disease in the same taluka."""
        return self.size >= CLUSTER_MIN_DETECTIONS


def cluster_detections(
    detections: list[Detection], now: datetime | None = None
) -> list[OutbreakCluster]:
    """Cluster same (pest_disease_id, taluka) detections chained by ≤7-day gaps.

    Callers pre-filter to the FR8 lookback window (now - 7d .. now); gaps longer
    than CLUSTER_WINDOW_DAYS split the chain into separate clusters. Returns
    clusters regardless of size — apply `.qualifies()` before notifying officers.

    ponytail: O(n²) worst-case same-bucket pairing is fine at demo volume;
    replace with a (taluka, pest_disease_id) DB index when the detections
    table exists.
    """
    del now  # window filtering is the caller's job; kept in the signature for the backend contract
    by_key: dict[tuple[int, str], list[Detection]] = defaultdict(list)
    for d in detections:
        by_key[(d.pest_disease_id, d.taluka)].append(d)

    clusters: list[OutbreakCluster] = []
    counter = itertools.count()
    for (pest_disease_id, taluka), dets in sorted(by_key.items()):
        dets.sort(key=lambda d: d.detected_at)
        current: OutbreakCluster | None = None
        last_seen: datetime | None = None
        for d in dets:
            chained = (
                current is not None
                and last_seen is not None
                and (d.detected_at - last_seen).days <= CLUSTER_WINDOW_DAYS
            )
            if not chained:
                current = OutbreakCluster(
                    cluster_id=f"OC-{next(counter):06d}",
                    pest_disease_id=pest_disease_id,
                    taluka=taluka,
                )
                clusters.append(current)
            assert current is not None
            current.detection_ids.append(d.detection_id)
            current.started_at = current.started_at or d.detected_at
            current.last_seen_at = d.detected_at
            current.size += 1
            last_seen = d.detected_at
    return clusters


# --------------------------------------------------------------------------- #
# NFR3: jurisdiction scoping
# --------------------------------------------------------------------------- #


def scope_by_jurisdiction(records: list, officer_taluka: str, taluka_key: str = "taluka") -> list:
    """NFR3: officer-facing queries are filtered at THIS seam, not in handlers.

    Dual models multiply officer-facing endpoints; every one of them must pass
    through this single filter so jurisdiction scoping can't be forgotten.

    ponytail: in-memory per-record check; swap for a SQL
    WHERE taluka = :officer_taluka when the backend query layer exists.
    """
    return [r for r in records if getattr(r, taluka_key, None) == officer_taluka]


# --------------------------------------------------------------------------- #
# FR9: audit log
# --------------------------------------------------------------------------- #


@dataclass
class AuditEntry:
    ts: datetime
    actor_id: str
    action: str
    entity: str
    detail: str = ""


class AuditLog:
    """FR9 stub: every officer/admin mutating action must land here.

    ponytail: in-memory, session-scoped; replace with the backend audit_log
    table (append-only). Never log image payloads or farmer PII in `detail`.
    """

    def __init__(self) -> None:
        self.entries: list[AuditEntry] = []

    def record(self, actor_id: str, action: str, entity: str, detail: str = "") -> AuditEntry:
        entry = AuditEntry(datetime.now(), actor_id, action, entity, detail)
        self.entries.append(entry)
        log.info("audit: actor=%s action=%s entity=%s %s", actor_id, action, entity, detail)
        return entry


# --------------------------------------------------------------------------- #
# FR7: IPM recommendation selection
# --------------------------------------------------------------------------- #


@dataclass
class IPMRecommendation:
    recommendation_id: str
    pest_disease_id: int
    severity_tier: str
    priority_rank: int
    method: str  # biological | cultural | chemical
    description: str = ""


def select_ipm_recommendations(
    catalog: list[IPMRecommendation], pest_disease_id: int, severity_tier: str
) -> list[IPMRecommendation]:
    """FR7: filter by (pest_disease_id, severity_tier), order by priority_rank.

    Seed-data contract: biological/cultural recommendations carry lower
    priority_rank values than chemical ones, so rank order alone puts
    biological/cultural before chemical. Enforce that contract in the seed
    script, not here.

    ponytail: in-memory filter over the seeded catalog; swap for the
    ipm_recommendations SQL query once the table exists.
    """
    matched = [
        r
        for r in catalog
        if r.pest_disease_id == pest_disease_id and r.severity_tier == severity_tier
    ]
    return sorted(matched, key=lambda r: r.priority_rank)


# --------------------------------------------------------------------------- #
# Regional history fusion input (FR4) + cold-start contract (Issue 5)
# --------------------------------------------------------------------------- #


def fetch_regional_history(
    detections: list[Detection],
    taluka: str,
    pest_disease_id: int,
    now: datetime | None = None,
    window_days: int = 7,
) -> int:
    """Count detections of this pest/disease in this taluka within the window.

    Cold-start contract (Issue 5): a return of 0 means "no regional evidence
    available", NEVER "no risk". Callers must surface that as a cold-start flag
    (see the service layer) — silently treating zero history as zero risk is a
    correctness bug, not a default.
    """
    now = now or datetime.now()
    cutoff = now.timestamp() - window_days * 24 * 3600
    return sum(
        1
        for d in detections
        if d.taluka == taluka
        and d.pest_disease_id == pest_disease_id
        and d.detected_at.timestamp() >= cutoff
    )
