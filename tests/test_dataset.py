"""Tests for dataset construction, splits, and determinism."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from cropguard.config import Config
from cropguard.data.dataset import (
    CropDiseaseDataModule,
    CropDiseaseDataset,
    load_split_lines,
    resolve_split_paths,
    _carve_val_from_train,
)
from cropguard.taxonomy import TAXONOMY


def test_resolve_split_paths_carves_val(tiny_cfg: Config):
    train_f, val_f = resolve_split_paths(tiny_cfg)
    assert train_f.exists() and val_f.exists()
    train_pairs = load_split_lines(train_f)
    val_pairs = load_split_lines(val_f)
    assert len(train_pairs) > 0 and len(val_pairs) > 0
    # disjoint
    train_set = {p for _, p in train_pairs}
    val_set = {p for _, p in val_pairs}
    assert not (train_set & val_set)


def test_carve_val_deterministic(tiny_cfg: Config):
    train_f, _ = resolve_split_paths(tiny_cfg)
    splits = train_f.parent
    a = _carve_val_from_train(tiny_cfg, train_f, splits, "color")
    b = _carve_val_from_train(tiny_cfg, train_f, splits, "color")
    assert a[0].read_text() == b[0].read_text()
    assert a[1].read_text() == b[1].read_text()


def test_dataset_returns_all_fields(tiny_cfg: Config):
    train_f, _ = resolve_split_paths(tiny_cfg)
    ds = CropDiseaseDataset(load_split_lines(train_f), image_size=32)
    assert len(ds) > 0
    item = ds[0]
    assert item["image"].shape == (3, 32, 32)
    assert 0 <= item["target"].item() < len(TAXONOMY.classes)
    assert item["crop_id"].dtype == torch.long
    assert item["weather"].shape == (4,)


def test_dataset_targets_match_class(tiny_cfg: Config):
    train_f, _ = resolve_split_paths(tiny_cfg)
    ds = CropDiseaseDataset(load_split_lines(train_f), image_size=32)
    for cls, path in ds.samples[:3]:
        idx = TAXONOMY.classes.index(cls)
        item = ds[ds.samples.index((cls, path))]
        assert idx == item["target"].item()


def test_subset_per_class(tiny_cfg: Config):
    train_f, _ = resolve_split_paths(tiny_cfg)
    full = CropDiseaseDataset(load_split_lines(train_f), image_size=32)
    sub = CropDiseaseDataset(
        load_split_lines(train_f), image_size=32, subset_per_class=2
    )
    # 4 classes; the full split keeps 1 val + 5 train per class here.
    assert 0 < len(sub) < len(full)
    assert len(sub) <= 2 * 4  # 2 per class × 4 classes


def test_datamodule_loaders(tiny_cfg: Config):
    dm = CropDiseaseDataModule(tiny_cfg)
    loaders = dm.loaders(batch_size=4)
    batch = next(iter(loaders["train"]))
    assert batch["image"].shape[0] <= 4
    assert batch["image"].shape[1] == 3
    assert batch["target"].shape == batch["crop_id"].shape
