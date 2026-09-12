"""Deterministic dataset split construction.

Protocol:
  1. Read the original PlantVillage train/test protocol (color_train.txt /
     color_test.txt) so the test set is the canonical held-out one.
  2. Validate every image; drop corrupt/missing files and record why.
  3. Compute perceptual hashes; detect near-duplicates. Reassign duplicate
     clusters so no near-identical image straddles train and validation/test.
  4. Carve a validation set from the train protocol (stratified by class,
     deterministic seed).
  5. Optionally cap per-class training samples.

Outputs a manifest CSV per split:
    split, image_path, class_name, crop_id, pest_disease_id, width, height, bytes
"""
from __future__ import annotations

import csv
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

from ..taxonomy import TAXONOMY
from .validation import dhash, find_near_duplicates, load_validated_image

log = logging.getLogger(__name__)


def read_protocol(root: Path, variant: str, split: str) -> list[str]:
    """Read color_train.txt / color_test.txt lines."""
    txt = root / "splits" / f"{variant}_{split}.txt"
    if not txt.exists():
        raise FileNotFoundError(f"Split protocol file missing: {txt}")
    lines = [ln.strip() for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return lines


def _resolve_path(root: Path, rel: str) -> Path:
    # protocol lines look like "raw/color/Class/file.JPG"
    return (root / rel).resolve()


def _class_name_from_path(rel: str) -> str:
    return Path(rel).parts[-2]


def build_manifest(
    root: Path,
    variant: str = "color",
    val_ratio: float = 0.2,
    train_subset_per_class: int | None = None,
    seed: int = 42,
    duplicate_threshold: float = 0.05,
    out_csv: Path | None = None,
) -> dict[str, list[dict]]:
    """Build deterministic train/val/test manifests. Returns {"train": [...], "val": [...], "test": [...]}."""
    rng = random.Random(seed)

    test_lines = read_protocol(root, variant, "test")
    train_lines = read_protocol(root, variant, "train")
    log.info("Protocol: %d train / %d test lines", len(train_lines), len(test_lines))

    # --- validate all images once -------------------------------------------------
    cache: dict[str, dict | None] = {}

    def validate(rel: str) -> dict | None:
        if rel not in cache:
            p = _resolve_path(root, rel)
            result = load_validated_image(p)
            if result is None:
                cache[rel] = None
            else:
                img, meta = result
                meta.update(
                    {
                        "rel": rel,
                        "class_name": _class_name_from_path(rel),
                        "hash": dhash(img),
                        "crop_id": TAXONOMY.crop_id(_class_name_from_path(rel)),
                        "pest_disease_id": TAXONOMY.pest_disease_id(_class_name_from_path(rel)),
                    }
                )
                cache[rel] = meta
        return cache[rel]

    test_items = [m for m in (validate(ln) for ln in test_lines) if m is not None]
    train_items = [m for m in (validate(ln) for ln in train_lines) if m is not None]
    rejected = sum(1 for ln in test_lines + train_lines if validate(ln) is None)
    if rejected:
        log.warning("Rejected %d corrupt/missing images during validation", rejected)

    # Detect cross-split near-duplicates (the real leakage vector).
    # Any near-dup pair spanning train<->test is a leakage hazard; we move the
    # test-side image into the train split so the held-out set stays clean.
    test_hashes = {t["hash"] for t in test_items}
    train_hashes = {t["hash"] for t in train_items}
    reassigned = 0
    # Pairwise bucket check using the shared 16-bit prefix buckets from both lists.
    combined = train_items + test_items
    pairs = find_near_duplicates(combined, threshold=duplicate_threshold)
    for i, j, _ in pairs:
        a, b = combined[i], combined[j]
        if (a["hash"] in train_hashes and b["hash"] in test_hashes) or (
            b["hash"] in train_hashes and a["hash"] in test_hashes
        ):
            # move the test-side member into train
            for m in (a, b):
                if m["hash"] in test_hashes:
                    test_hashes.discard(m["hash"])
                    train_hashes.add(m["hash"])
                    test_items.remove(m)
                    train_items.append(m)
                    reassigned += 1
    if reassigned:
        log.warning(
            "Moved %d near-duplicate images from test -> train to prevent leakage", reassigned
        )

    # --- carve validation from train, stratified by class ---------------------------
    by_class: dict[str, list[dict]] = defaultdict(list)
    for item in train_items:
        by_class[item["class_name"]].append(item)

    train_final: list[dict] = []
    val_final: list[dict] = []
    for cls, members in by_class.items():
        rng.shuffle(members)
        n_val = max(1, int(round(len(members) * val_ratio)))
        val_final.extend(members[:n_val])
        train_final.extend(members[n_val:])

    # --- optional per-class cap on training (keeps dev runs fast on CPU) ------------
    if train_subset_per_class is not None:
        capped: list[dict] = []
        cap_by_class: dict[str, list[dict]] = defaultdict(list)
        for item in train_final:
            cap_by_class[item["class_name"]].append(item)
        for cls, members in cap_by_class.items():
            rng.shuffle(members)
            capped.extend(members[:train_subset_per_class])
        train_final = capped
        log.info("Per-class cap %d applied → train=%d", train_subset_per_class, len(train_final))

    rng.shuffle(train_final)
    rng.shuffle(val_final)
    rng.shuffle(test_items)

    manifests = {
        "train": _finalize(train_final, "train"),
        "val": _finalize(val_final, "val"),
        "test": _finalize(test_items, "test"),
    }
    log.info(
        "Manifest sizes: train=%d val=%d test=%d",
        len(manifests["train"]),
        len(manifests["val"]),
        len(manifests["test"]),
    )

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "split", "rel", "class_name", "crop_id", "pest_disease_id",
                    "width", "height", "bytes",
                ],
            )
            writer.writeheader()
            for split, items in manifests.items():
                for item in items:
                    writer.writerow(
                        {
                            "split": split,
                            "rel": item["rel"],
                            "class_name": item["class_name"],
                            "crop_id": item["crop_id"],
                            "pest_disease_id": item["pest_disease_id"],
                            "width": item["width"],
                            "height": item["height"],
                            "bytes": item["bytes"],
                        }
                    )
        log.info("Wrote manifest to %s", out_csv)
    return manifests


def _finalize(items: list[dict], split: str) -> list[dict]:
    out = []
    for item in items:
        out.append(
            {
                "split": split,
                "rel": item["rel"],
                "class_name": item["class_name"],
                "crop_id": item["crop_id"],
                "pest_disease_id": item["pest_disease_id"],
                "width": item["width"],
                "height": item["height"],
                "bytes": item["bytes"],
            }
        )
    return out


def class_distribution(items: list[dict]) -> Counter:
    return Counter(i["class_name"] for i in items)