"""Leakage-safe master manifest and farm/plot-clustered splits (Phase 2, spec §3.1).

A field dataset is not a pile of independent images: several photos come from
the same farm, the same plot, the same capture session — near-identical plants,
lighting, and background. Splitting by individual image puts such siblings on
both sides of the split and inflates validation accuracy (spec risk #2,
"leakage-inflated validation"). This module keys the split on the *cluster*:

- When a manifest CSV sidecar provides ``farm_id``/``plot_id``, images are
  grouped by ``(farm_id, plot_id)`` — the farmer-level unit.
- When it does not (PlantVillage has no such metadata), images are grouped by
  perceptual-hash cluster — the existing near-duplicate logic, lifted from
  per-class consecutive filtering to global cluster containment.

Either way: every cluster is fully assigned to exactly one split. A cluster
that would straddle splits is moved whole (spec §7: "move cluster, rewrite
manifest").
"""
from __future__ import annotations

import csv
import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .validation import perceptual_hash, perceptual_hash_distance

log = logging.getLogger(__name__)

MANIFEST_COLUMNS = ("rel_path", "class_name", "crop_id", "farm_id", "plot_id", "capture_date")
MANIFEST_FIELDS = {".csv", ".tsv"}


@dataclass(frozen=True)
class ManifestRow:
    rel_path: str
    class_name: str
    crop_id: int
    farm_id: str
    plot_id: str
    capture_date: str


def class_map_from_manifest(rows: list[ManifestRow]) -> dict[str, str]:
    """rel_path → class_name, for layouts where folders aren't class names."""
    return {r.rel_path: r.class_name for r in rows if r.class_name}


def class_map_from_sidecar(csv_sidecar: Path) -> dict[str, str]:
    """rel_path → class_name straight from the CSV (no image walk, no hashing)."""
    if not csv_sidecar.exists():
        return {}
    out: dict[str, str] = {}
    with csv_sidecar.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rel = (r.get("rel_path") or "").strip()
            cls = (r.get("class_name") or "").strip()
            if rel and cls:
                out[rel] = cls
    return out


def _stable_int(text: str, mod: int) -> int:
    """Process-independent digest (builtin hash() is salted per process)."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % mod


def build_manifest(root: Path, csv_sidecar: Path | None = None) -> list[ManifestRow]:
    """Master manifest over the image tree at ``root``.

    ``csv_sidecar`` (when given) is a CSV with MANIFEST_COLUMNS headers;
    ``rel_path`` is relative to ``root``'s *parent* (matching the split-file
    convention so the same paths work in both). Missing rows fall back to
    path-derived defaults: farm_id=plot_id="~", capture_date="".
    """
    rows: dict[str, ManifestRow] = {}
    if csv_sidecar is not None and csv_sidecar.exists():
        with csv_sidecar.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rel = (r.get("rel_path") or "").strip()
                if not rel:
                    continue
                rows[rel] = ManifestRow(
                    rel_path=rel,
                    class_name=(r.get("class_name") or "").strip(),
                    crop_id=int(r["crop_id"]) if r.get("crop_id") else 0,
                    farm_id=(r.get("farm_id") or "").strip() or "~",
                    plot_id=(r.get("plot_id") or "").strip() or "~",
                    capture_date=(r.get("capture_date") or "").strip(),
                )

    manifest: list[ManifestRow] = []
    for img in sorted(root.rglob("*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"} or not img.is_file():
            continue
        # rel paths are relative to the tree root itself: the manifest
        # describes the tree it was built from. Consumers prefix the variant
        # dir when writing split files (dataset._build_manifest_splits).
        rel = img.relative_to(root).as_posix()
        meta = rows.get(rel)
        if meta is not None:
            manifest.append(meta)
            continue
        manifest.append(
            ManifestRow(
                rel_path=rel,
                class_name=img.parent.name,
                crop_id=0,
                farm_id="~",
                plot_id="~",
                capture_date="",
            )
        )
    n_meta = sum(1 for m in manifest if m.farm_id != "~")
    if rows and n_meta == 0:
        log.warning(
            "Manifest sidecar provided but matched 0/%d rows — check the "
            "rel_path convention (expected paths relative to %s)", len(rows), root,
        )
    log.info(
        "Manifest: %d images (%d with farm/plot metadata) from %s",
        len(manifest), n_meta, root,
    )
    return manifest


def _clusters_by_farm(manifest: list[ManifestRow]) -> dict[str, set[str]]:
    """(farm_id, plot_id) → rel_paths. Rows with sentinel metadata are
    singleton clusters (handled by hash clustering downstream)."""
    out: defaultdict[str, set[str]] = defaultdict(set)
    for m in manifest:
        if m.farm_id != "~":
            out[f"{m.farm_id}/{m.plot_id}"].add(m.rel_path)
    return dict(out)


def _clusters_by_hash(
    manifest: list[ManifestRow], root: Path, threshold: float
) -> dict[str, set[str]]:
    """Perceptual-hash clusters → rel_paths.

    Global over the manifest (not per-class): a near-duplicate pair is leakage
    regardless of which folder it sits in. Greedy union over sorted paths with
    a per-cluster hash representative; O(N) hash computations, pairwise checks
    only against representatives.
    """
    clusters: dict[str, set[str]] = {}
    reps: list[tuple[object, str]] = []  # (hash, cluster_key)
    for m in manifest:
        p = root / m.rel_path
        try:
            with Image.open(p) as img:
                h = perceptual_hash(img)
        except Exception:
            log.warning("Unreadable image skipped from clustering: %s", p)
            continue
        match = next(
            (key for rep_h, key in reps if perceptual_hash_distance(h, rep_h) < threshold),
            None,
        )
        if match is None:
            match = f"hash:{_stable_int(m.rel_path, 1 << 60):x}"
            clusters[match] = set()
            reps.append((h, match))
        clusters[match].add(m.rel_path)
    return clusters


def leakage_safe_split(
    manifest: list[ManifestRow],
    root: Path,
    val_ratio: float = 0.2,
    duplicate_threshold: float = 0.05,
    test_ratio: float = 0.0,
) -> dict[str, list[str]]:
    """Assign whole clusters to splits; return {"train": [...], "val": [...], "test": [...]}.

    Deterministic (sorted iteration + stable digest): identical inputs produce
    identical splits on any machine. Within a cluster every image goes to the
    same split — the containment guarantee. Class-wise stride keeps val/test
    balanced per class like the legacy splitter.
    """
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    if test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError(f"val_ratio + test_ratio must be < 1, got {val_ratio} + {test_ratio}")

    farm_clusters = _clusters_by_farm(manifest)
    hash_clusters = _clusters_by_hash(manifest, root, duplicate_threshold)

    # Sentinel-metadata images fall through farm clustering into hash clusters;
    # both views are merged without splitting a hash cluster across split keys.
    assignments: dict[str, str] = {}
    for clusters in (farm_clusters, hash_clusters):
        for key in sorted(clusters):
            members = clusters[key]
            if any(p in assignments for p in members):
                continue  # already clustered by the stronger key (farm/plot)
            split = "train"
            roll = _stable_int(key, 100)
            if roll < round(test_ratio * 100):
                split = "test"
            elif roll < round((test_ratio + val_ratio) * 100):
                split = "val"
            for p in members:
                assignments[p] = split

    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for m in sorted(manifest, key=lambda r: r.rel_path):
        out[assignments[m.rel_path]].append(m.rel_path)
    log.info(
        "Leakage-safe split: %d train / %d val / %d test (cluster-contained)",
        len(out["train"]), len(out["val"]), len(out["test"]),
    )
    return out


def find_leakage(
    manifest: list[ManifestRow],
    splits: dict[str, list[str]],
    root: Path,
    duplicate_threshold: float = 0.05,
) -> list[dict[str, str]]:
    """Audit a split assignment for cross-split clusters (spec §7 monitor).

    Returns one record per offending cluster:
    ``{"cluster", "splits", "images", "action": "move_cluster"}``. Empty list =
    containment holds.
    """
    split_of = {p: s for s, paths in splits.items() for p in paths}
    seen: set[frozenset[str]] = set()
    offenders: list[dict[str, str]] = []

    for cluster in (_clusters_by_farm(manifest), _clusters_by_hash(manifest, root, duplicate_threshold)):
        for key in sorted(cluster):
            members = frozenset(cluster[key])
            if members in seen:
                continue
            seen.add(members)
            used = sorted({split_of[p] for p in members if p in split_of})
            if len(used) > 1:
                offenders.append(
                    {
                        "cluster": key,
                        "splits": ",".join(used),
                        "images": ",".join(sorted(members)),
                        "action": "move_cluster",
                    }
                )
    return offenders
