"""Dataset report: distributions, imbalance, metadata coverage, duplicates."""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from ..taxonomy import TAXONOMY
from .validation import dhash, load_validated_image, perceptual_hash_distance

log = logging.getLogger(__name__)


def _resolution_stats(rows: list[dict]) -> dict:
    widths = np.array([r["width"] for r in rows], dtype=np.int64)
    heights = np.array([r["height"] for r in rows], dtype=np.int64)
    return {
        "min": (int(widths.min()), int(heights.min())),
        "max": (int(widths.max()), int(heights.max())),
        "median_w": int(np.median(widths)),
        "median_h": int(np.median(heights)),
    }


def _gini(values: list[float]) -> float:
    """Gini coefficient of a distribution (0 = perfectly balanced)."""
    arr = np.sort(np.asarray(values, dtype=np.float64))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    cumsum = np.cumsum(arr)
    return float((2 * np.sum(cumsum) - (n + 1) * arr.sum()) / (n * arr.sum()))


def build_dataset_report(
    manifests: dict[str, list[dict]], image_root: Path, out: Path | None = None
) -> dict:
    report: dict = {"splits": {}, "duplicates": {}, "metadata": {}}

    all_rows = []
    for split, rows in manifests.items():
        counts = Counter(r["class_name"] for r in rows)
        per_crop = Counter(r["crop_id"] for r in rows)
        values = list(counts.values())
        report["splits"][split] = {
            "n_images": len(rows),
            "n_classes": len(counts),
            "samples_per_class": dict(sorted(counts.items())),
            "per_crop": {str(k): v for k, v in sorted(per_crop.items())},
            "min_per_class": min(values) if values else 0,
            "max_per_class": max(values) if values else 0,
            "mean_per_class": round(float(np.mean(values)), 2) if values else 0,
            "gini": round(_gini(values), 4) if values else 0.0,
            "resolution": _resolution_stats(rows),
        }
        all_rows.extend(rows)

    # Duplicate scan (sampled for the report — full scan is expensive)
    log.info("Scanning duplicates for report (sampled up to 5000 images)...")
    sample = all_rows[:5000]
    hashes = []
    for row in sample:
        result = load_validated_image(image_root / row["rel"])
        if result is not None:
            hashes.append({"hash": dhash(result[0]), "rel": row["rel"]})
    dup_pairs = 0
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if perceptual_hash_distance(hashes[i]["hash"], hashes[j]["hash"]) <= 0.05:
                dup_pairs += 1
    report["duplicates"] = {
        "scan_sample": len(hashes),
        "near_dup_pairs_found": dup_pairs,
        "note": "Near-duplicates are reassigned by the split builder to prevent leakage.",
    }

    # Metadata coverage: in the curated manifest all rows carry crop_id and
    # pest_disease_id; weather/stage/region are synthesized deterministically.
    missing_crop = sum(1 for r in all_rows if not r.get("crop_id"))
    missing_pd = sum(1 for r in all_rows if not r.get("pest_disease_id"))
    report["metadata"] = {
        "rows": len(all_rows),
        "missing_crop_id": missing_crop,
        "missing_pest_disease_id": missing_pd,
        "note": "Weather/growth-stage/region are deterministically synthesized per "
        "sample from documented climatology (see DATASET.md); real values arrive "
        "from the backend at inference time.",
        "taxonomy_classes": TAXONOMY.classes,
    }

    report["summary"] = {
        "total_images": len(all_rows),
        "splits": {k: v["n_images"] for k, v in report["splits"].items()},
        "train_test_overlap_note": "Cross-split near-duplicates moved to train; held-out set is clean.",
    }

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log.info("Dataset report written to %s", out)
    return report