"""Phase 2 (spec §3.1): leakage-safe farm/plot-clustered splits.

The spec's acceptance test: a cluster (same farm/plot, or perceptual near-
duplicate) must land fully in ONE split — never straddle train/val.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from cropguard.config import Config
from cropguard.data.dataset import CropDiseaseDataModule, resolve_split_paths
from cropguard.data.manifest import (
    ManifestRow,
    build_manifest,
    find_leakage,
    leakage_safe_split,
)

# Real taxonomy classes: the dataset maps class_name → index via TAXONOMY,
# so fixtures must use actual label-space members (two different crops).
CLS_A, CLS_B = "Tomato___Late_blight", "Potato___Late_blight"


def _sidecar_lines() -> list[str]:
    """rel_path convention: relative to the image-tree root (field/), matching
    manifest.build_manifest."""
    lines = ["rel_path,class_name,crop_id,farm_id,plot_id,capture_date"]
    for cls in (CLS_A, CLS_B):
        for f in range(3):
            for p in range(4):
                for i in range(5):
                    lines.append(
                        f"farm{f}_plot{p}_{cls}/img_{i}.jpg,{cls},0,farm{f},plot{p},2026-09-01"
                    )
    return lines


def _write_sidecar(field_root: Path) -> Path:
    csv_sidecar = field_root / "manifest.csv"
    csv_sidecar.write_text("\n".join(_sidecar_lines()) + "\n", encoding="utf-8")
    return csv_sidecar


@pytest.fixture
def field_root(tmp_path: Path) -> Path:
    """Synthetic field dataset: 2 crops × 3 farms × 4 plots × 5 images.

    Images within a (farm, plot) share one texture (same perceptual hash → one
    hash cluster per plot); different plots get different textures, the way
    real field photos from different farms differ.
    """
    import numpy as np

    rng = np.random.default_rng(42)
    root = tmp_path / "field"
    for cls in (CLS_A, CLS_B):
        for f in range(3):
            for p in range(4):
                plot_dir = root / f"farm{f}_plot{p}_{cls}"
                plot_dir.mkdir(parents=True, exist_ok=True)
                # One texture per plot: noisy but stable within the plot.
                base = rng.integers(0, 255, size=(48, 48, 3), dtype=np.uint8)
                tinted = np.clip(
                    base.astype(int)
                    + np.array([90 + f * 10, 140, 60 + p * 8], dtype=int),
                    0,
                    255,
                ).astype(np.uint8)
                texture = Image.fromarray(tinted)
                for i in range(5):
                    texture.save(plot_dir / f"img_{i}.jpg", quality=95)
    return root


def _rows(field_root: Path, farms: bool) -> list[ManifestRow]:
    csv_sidecar = _write_sidecar(field_root) if farms else None
    return build_manifest(field_root, csv_sidecar)


# --------------------------------------------------------------------------- #
# Containment: the spec's core guarantee
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("farms", [True, False], ids=["farm_key", "hash_key"])
def test_plot_cluster_never_straddles_splits(field_root: Path, farms: bool):
    manifest = _rows(field_root, farms)
    splits = leakage_safe_split(manifest, field_root, val_ratio=0.25)

    plot_of = {}
    for m in manifest:
        plot_of[m.rel_path] = m.rel_path.split("_img")[0]  # farm{f}_plot{p}_{cls}

    for plot in set(plot_of.values()):
        members = [p for p, pl in plot_of.items() if pl == plot]
        used = {s for s in ("train", "val", "test") for p in members if p in splits[s]}
        assert len(used) == 1, f"plot {plot} straddles splits: {used}"

    n = sum(len(v) for v in splits.values())
    assert n == len(manifest) == 2 * 3 * 4 * 5


def test_duplicate_pair_lands_in_one_split(tmp_path: Path):
    """Near-duplicate images in DIFFERENT folders are still one cluster."""
    root = tmp_path / "dup"
    a, b = root / "class_a", root / "class_b"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    Image.new("RGB", (64, 64), color=(120, 130, 90)).save(a / "a1.jpg")
    Image.new("RGB", (64, 64), color=(120, 130, 90)).save(b / "b1.jpg")  # identical
    for i in range(8):  # filler so class_a has enough volume to split
        Image.new("RGB", (64, 64), color=(10 * i + 5, 200, 30)).save(a / f"f{i}.jpg")

    manifest = build_manifest(root, None)
    splits = leakage_safe_split(manifest, root, val_ratio=0.3)

    loc = {p.split("/")[-1]: s for s, paths in splits.items() for p in paths}
    assert loc["a1.jpg"] == loc["b1.jpg"], "near-duplicates straddled the split"


def test_find_leakage_detects_and_reports(field_root: Path):
    manifest = _rows(field_root, farms=True)
    splits = leakage_safe_split(manifest, field_root, val_ratio=0.25)

    assert find_leakage(manifest, splits, field_root) == []  # built split is clean

    # Force a violation: move one image of some fully-train plot into val.
    victim = splits["train"][0]
    plot_dir = victim.rsplit("/", 1)[0]  # farm{f}_plot{p}_{cls}
    farm, plot = plot_dir.split("_")[0], plot_dir.split("_")[1]
    splits["train"].remove(victim)
    splits["val"].append(victim)
    offenders = find_leakage(manifest, splits, field_root)
    assert offenders and all(o["action"] == "move_cluster" for o in offenders)
    assert any(f"{farm}/{plot}" in o["cluster"] for o in offenders)


def test_split_is_deterministic(field_root: Path):
    manifest = _rows(field_root, farms=True)
    s1 = leakage_safe_split(manifest, field_root, val_ratio=0.25)
    s2 = leakage_safe_split(manifest, field_root, val_ratio=0.25)
    assert s1 == s2


# --------------------------------------------------------------------------- #
# Integration with the datamodule
# --------------------------------------------------------------------------- #


def test_resolve_split_paths_autodetects_sidecar(field_root: Path):
    _write_sidecar(field_root)

    cfg = Config()
    cfg.data.root = field_root
    train_f, val_f = resolve_split_paths(cfg)
    assert train_f.name.endswith("_manifest.txt") and val_f.name.endswith("_manifest.txt")

    # Containment holds through the whole resolve path: every (farm, plot)
    # cluster lands whole in one split (different plots of one farm may
    # legitimately differ — the spec's cluster unit is farm+plot).
    split_of = {}
    for name, path in (("train", train_f), ("val", val_f)):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                split_of[line.strip()] = name
    plots = {p.rsplit("_img", 1)[0] for p in split_of}
    for plot in plots:
        used = {s for p, s in split_of.items() if p.rsplit("_img", 1)[0] == plot}
        assert len(used) == 1, f"plot {plot} straddles {used}"


def test_loaders_accept_split_manifest(field_root: Path):
    cfg = Config()
    cfg.data.root = field_root
    cfg.data.workers = 0
    cfg.data.min_samples_per_class = 1
    cfg.data.image_size = 48
    _write_sidecar(field_root)

    manifest = build_manifest(field_root, field_root / "manifest.csv")
    splits = leakage_safe_split(manifest, field_root, val_ratio=0.25)

    dm = CropDiseaseDataModule(cfg)
    loaders = dm.loaders(batch_size=8, split_manifest=splits)
    batch = next(iter(loaders["train"]))
    assert batch["image"].shape[1:] == (3, 48, 48)
    assert batch["target"].shape[0] == batch["image"].shape[0]
