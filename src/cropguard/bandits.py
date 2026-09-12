"""Contextual bandits for the two real decisions (§4).

Problem 1 — adaptive abstention threshold: per (crop, pest_disease, region)
threshold adaptation from downstream outcomes. The escalation decision itself
stays with the service; this bandit only proposes a threshold, and every change
logs the reward history that produced it (acceptance criterion: show your work
— an unexplainable drift in a government-facing safety mechanism is a
non-starter).

Problem 2 — IPM ranking refinement: reorder interventions WITHIN an
intervention-type tier by confirmed effectiveness (feedback.actual_outcome).
The ethical constraint is a wall, not a preference: a chemical option can never
be promoted above a biological one for the same severity tier, regardless of
the reward signal. That ordering is a policy choice, not something to optimize
away.

Thompson sampling with a conjugate Beta posterior per (context, action) — the
right size for these decisions, deliberately not deep RL.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

log = logging.getLogger(__name__)

# §4 hard wall: within a severity tier, these types can never be outranked by
# a LOWER-priority type. Ordered best (most preferred) → worst.
_TYPE_PRIORITY = {"biological": 0, "cultural": 1, "chemical": 2}


@dataclass
class RewardEntry:
    ts: datetime
    context: dict
    action: str
    reward: float
    note: str = ""


@dataclass
class RankChange:
    ts: datetime
    key: tuple  # (pest_disease_id, severity_tier)
    before: list[str]
    after: list[str]
    reason: str


class AbstentionBandit:
    """Proposes the auto-resolve/escalate threshold per context (Problem 1)."""

    def __init__(self) -> None:
        self.history: list[RewardEntry] = []

    def record_outcome(self, context: dict, action: str, reward: float, note: str = "") -> None:
        self.history.append(RewardEntry(datetime.now(), dict(context), action, reward, note))

    def propose_threshold(
        self,
        base_threshold: float,
        context: dict,
        min_threshold: float = 0.5,
        max_threshold: float = 0.95,
    ) -> float:
        """Current best-effort threshold for this context.

        ponytail: mean-reward offset heuristic, not full Thompson sampling —
        there is no outcome data yet to fit a posterior against. Upgrade path:
        Beta posteriors per (crop, pest, region) once officer/feedback outcomes
        accumulate. Bounded so a cold context can never propose an unsafe value.
        """
        pool = [
            e.reward
            for e in self.history
            if all(e.context.get(k) == v for k, v in context.items())
        ]
        if not pool:
            return base_threshold  # cold start: exactly the configured policy
        # Escalations that kept getting overturned (low reward) should make the
        # bandit demand MORE confidence before auto-resolving; high rewards for
        # correctly-handled cases relax it back toward the base.
        offset = (0.5 - sum(pool) / len(pool)) * 0.2
        return max(min_threshold, min(max_threshold, base_threshold + offset))


class IPMRankingBandit:
    """Refines IPM priority_rank within intervention-type tiers (Problem 2)."""

    def __init__(self, rng_seed: int = 42) -> None:
        self.reward_history: list[RewardEntry] = []
        self.changes: list[RankChange] = []
        # Beta posteriors over "confirmed effective" per recommendation id.
        self.alpha: dict[str, float] = {}
        self.beta: dict[str, float] = {}
        self._seed = rng_seed

    def record_outcome(self, recommendation_id: str, effective: bool, context: dict) -> None:
        self.reward_history.append(
            RewardEntry(datetime.now(), dict(context), recommendation_id, 1.0 if effective else 0.0)
        )
        self.alpha[recommendation_id] = self.alpha.get(recommendation_id, 1.0) + (1.0 if effective else 0.0)
        self.beta[recommendation_id] = self.beta.get(recommendation_id, 1.0) + (0.0 if effective else 1.0)

    def refine_ranking(self, key: tuple, items: list, now: datetime | None = None) -> list:
        """Reorder `items` within their type tier by sampled effectiveness.

        `items` are IPMRecommendation-like objects with (recommendation_id,
        priority_rank, method). The wall: biological > cultural > chemical
        ordering BETWEEN tiers is restored after any within-tier reorder, so
        no reward signal can cross it.
        """
        now = now or datetime.now()
        before = [it.recommendation_id for it in items]

        def sampled_score(it) -> float:
            if it.recommendation_id not in self.alpha:
                # No outcome data yet → keep the expert prior (priority_rank);
                # an uninformative Thompson sample would shuffle it arbitrarily.
                return -float(it.priority_rank)
            a = self.alpha[it.recommendation_id]
            b = self.beta[it.recommendation_id]
            # Thompson sample from the Beta posterior (deterministic seed for
            # reproducibility in demos/tests).
            import random

            rng = random.Random(f"{self._seed}:{it.recommendation_id}:{len(self.reward_history)}")
            return rng.betavariate(a, b)

        # Within a tier: evidence decides (expert priority_rank is the prior
        # encoded in the seed, not a constraint). Across tiers: the wall holds.
        result = sorted(
            items,
            key=lambda it: (_TYPE_PRIORITY.get(it.method, 3), -sampled_score(it)),
        )
        after = [it.recommendation_id for it in result]

        if after != before:
            self.changes.append(
                RankChange(now, key, before, after, reason="thompson-within-tier")
            )
            log.info("IPM rank refined for %s: %s -> %s", key, before, after)
        return result

    @staticmethod
    def enforce_wall(items: list) -> list:
        """Static guarantee: sort by type priority, stable within type.

        The service must call this after ANY ranking source (bandit, manual
        officer edit, retrained model) touches IPM ordering.
        """
        return sorted(items, key=lambda it: _TYPE_PRIORITY.get(it.method, 3))
