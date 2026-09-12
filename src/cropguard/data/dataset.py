"""Deterministic PlantVillage dataset and splits.

Class order is fixed (sorted PlantVillage class names = `TAXONOMY.classes`) so
that every checkpoint, split file, and inference run shares one label space.

The val set is carved deterministically from the official train protocol file
(sorted per-class, every Nth sample) — no RNG, so splits are reproducible on
any machine regardless of library versions.
"""
from __future__ import annotations

import hashlib
import logging
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..config import Config, PROJECT_ROOT
from ..taxonomy import GROWTH_STAGES, REGIONS, TAXONOMY
from .manifest import class_map_from_sidecar
from .metadata import synthesize_weather
from .transforms import eval_transform, train_transform
from .validation import perceptual_hash, perceptual_hash_distance

log = logging.getLogger(__name__)

VARIANT_DIRS = {"color": "color", "grayscale": "grayscale", "segmented": "segmented"}

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def class_to_index(class_name: str) -> int:
    return TAXONOMY.classes.index(class_name)


def discover_classes(root: Path) -> list[str]:
    """Sorted class folder names under the variant dir."""
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not classes:
        raise FileNotFoundError(f"No class folders under {root}")
    return classes


def list_images(root: Path, class_name: str) -> list[Path]:
    d = root / class_name
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def build_split_files(
    root: Path,
    out_dir: Path,
    val_ratio: float = 0.20,
    min_samples_per_class: int = 30,
    duplicate_threshold: float = 0.05,
) -> tuple[Path, Path]:
    """Build train/val split files deterministically from the class folders.

    Per class: sort paths, drop consecutive near-duplicates via perceptual
    hash, then take every round(1/val_ratio)-th (sorted) sample for val.
    Writes <variant>_train.txt / <variant>_val.txt with root-relative POSIX
    paths and returns their paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    variant = root.name
    stride = max(2, round(1 / val_ratio)) if 0 < val_ratio < 1 else None

    train_lines: list[str] = []
    val_lines: list[str] = []
    for class_name in discover_classes(root):
        paths = list_images(root, class_name)
        if len(paths) < min_samples_per_class:
            log.warning("Skipping class %s: only %d images", class_name, len(paths))
            continue

        kept: list[Path] = []
        prev_hash = None
        for p in paths:
            try:
                with Image.open(p) as img:
                    h = perceptual_hash(img)
            except Exception:
                log.warning("Unreadable image skipped: %s", p)
                continue
            if prev_hash is not None and perceptual_hash_distance(h, prev_hash) < duplicate_threshold:
                continue
            kept.append(p)
            prev_hash = h

        n_val = max(1, round(len(kept) * val_ratio))
        for i, p in enumerate(kept):
            rel = p.relative_to(root.parent).as_posix()
            is_val = (i % stride == 0) if stride else (i == 0)
            if is_val:
                val_lines.append(rel)
            else:
                train_lines.append(rel)

    train_path = out_dir / f"{variant}_train.txt"
    val_path = out_dir / f"{variant}_val.txt"
    train_path.write_text("\n".join(sorted(train_lines)) + "\n", encoding="utf-8")
    val_path.write_text("\n".join(sorted(val_lines)) + "\n", encoding="utf-8")
    log.info("Wrote splits: %d train / %d val -> %s", len(train_lines), len(val_lines), out_dir)
    return train_path, val_path


def _resolve_plant_root(cfg: Config) -> tuple[Path, Path]:
    """(plant_root, variant_root) for cfg.data.root.

    Accepts .../PlantVillage, .../PlantVillage/color, or a root that directly
    holds the class folders.
    """
    variant = VARIANT_DIRS[cfg.data.variant]
    root = Path(cfg.data.root)
    if (root / variant).is_dir():
        plant_root = root
    elif (root.parent / variant).is_dir():
        plant_root = root.parent
    else:
        plant_root = root  # root itself holds the class folders
    variant_root = plant_root / variant
    if not variant_root.is_dir():
        variant_root = root
    return plant_root, variant_root


def _find_manifest_csv(plant_root: Path) -> Path | None:
    """Manifest CSV sidecar (Phase 2): conventional locations, else None."""
    for candidate in (plant_root / "manifest.csv", plant_root / "splits" / "manifest.csv"):
        if candidate.exists():
            return candidate
    return None


def _build_manifest_splits(
    cfg: Config,
    plant_root: Path,
    variant_root: Path,
    csv_sidecar: Path,
    splits_dir: Path,
    variant: str,
) -> tuple[Path, Path]:
    """Farm/plot-clustered leakage-safe splits from the manifest sidecar."""
    from .manifest import build_manifest, leakage_safe_split
    manifest = build_manifest(variant_root, csv_sidecar)
    if not manifest:
        log.warning("Manifest %s is empty — falling back to legacy split", csv_sidecar.name)
        return build_split_files(
            variant_root,
            splits_dir,
            val_ratio=cfg.data.val_ratio,
            min_samples_per_class=cfg.data.min_samples_per_class,
            duplicate_threshold=cfg.data.duplicate_threshold,
        )
    splits = leakage_safe_split(
        manifest,
        variant_root,
        val_ratio=cfg.data.val_ratio,
        duplicate_threshold=cfg.data.duplicate_threshold,
    )
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_p = splits_dir / f"{variant}_train_manifest.txt"
    val_p = splits_dir / f"{variant}_val_manifest.txt"
    # Manifest rel paths are variant-root-relative; when the tree has a variant
    # layer (PlantVillage/color), split lines carry the prefix, else not.
    prefix = f"{variant}/" if variant_root != plant_root else ""
    train_p.write_text(
        "\n".join(f"{prefix}{p}" for p in splits["train"]) + "\n", encoding="utf-8"
    )
    val_p.write_text(
        "\n".join(f"{prefix}{p}" for p in splits["val"]) + "\n", encoding="utf-8"
    )
    log.info(
        "Wrote manifest splits: %d train / %d val -> %s",
        len(splits["train"]), len(splits["val"]), splits_dir,
    )
    return train_p, val_p


def resolve_split_paths(cfg: Config) -> tuple[Path, Path]:
    """Locate train/val split files; carve or build them when missing."""
    variant = VARIANT_DIRS[cfg.data.variant]
    plant_root, variant_root = _resolve_plant_root(cfg)
    splits_dir = plant_root / "splits"

    train_f = splits_dir / f"{variant}_train.txt"
    val_f = splits_dir / f"{variant}_val.txt"
    if train_f.exists() and val_f.exists():
        return train_f, val_f

    if train_f.exists():
        # Official download ships train/test only → carve val out of train.
        log.info("No %s — carving val from %s", val_f.name, train_f.name)
        return _carve_val_from_train(cfg, train_f, splits_dir, variant)

    # Phase 2: a manifest CSV sidecar switches splitting to farm/plot-clustered
    # leakage-safe mode (backward compatible: absent sidecar = legacy behavior).
    csv_sidecar = _find_manifest_csv(plant_root)
    if csv_sidecar is not None:
        log.info("Manifest sidecar %s — building leakage-safe (farm/plot-clustered) splits", csv_sidecar)
        return _build_manifest_splits(cfg, plant_root, variant_root, csv_sidecar, splits_dir, variant)

    # No split files at all → build them from the class folders.
    log.info("No split files found — building deterministically from %s", variant_root)
    return build_split_files(
        variant_root,
        splits_dir,
        val_ratio=cfg.data.val_ratio,
        min_samples_per_class=cfg.data.min_samples_per_class,
        duplicate_threshold=cfg.data.duplicate_threshold,
    )


def _carve_val_from_train(
    cfg: Config, train_f: Path, splits_dir: Path, variant: str
) -> tuple[Path, Path]:
    """Deterministically move every Nth (per class, sorted) train line to val."""
    by_class: dict[str, list[str]] = {}
    for line in train_f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            by_class.setdefault(Path(line).parent.name, []).append(line)

    stride = max(2, round(1 / cfg.data.val_ratio))
    train_out: list[str] = []
    val_out: list[str] = []
    for cls in sorted(by_class):
        for i, line in enumerate(sorted(by_class[cls])):
            (val_out if i % stride == 0 else train_out).append(line)

    train_p = splits_dir / f"{variant}_train_carved.txt"
    val_p = splits_dir / f"{variant}_val_carved.txt"
    train_p.write_text("\n".join(train_out) + "\n", encoding="utf-8")
    val_p.write_text("\n".join(val_out) + "\n", encoding="utf-8")
    log.info("Carved splits: %d train / %d val", len(train_out), len(val_out))
    return train_p, val_p


def load_split_lines(path: Path) -> list[tuple[str, str]]:
    """Read a split file → [(class_name, absolute_path)], skipping missing files.

    Paths in the file may be relative to the splits' PlantVillage root, to the
    project root, or absolute. Manifest-generated splits (*_manifest.txt) get
    their class names from the manifest sidecar (folders there are farm/plot
    ids, not classes).
    """
    plant_root = path.parent.parent  # splits/…txt → PlantVillage root
    class_map = None
    variant_prefix = ""
    if path.name.endswith("_manifest.txt"):
        variant = path.name.split("_")[0]  # color_train_manifest.txt → color
        variant_prefix = f"{variant}/"
        csv_sidecar = _find_manifest_csv(plant_root)
        if csv_sidecar is not None:
            from .manifest import class_map_from_sidecar

            class_map = class_map_from_sidecar(csv_sidecar)
    return _resolve_split_lines(
        path.read_text(encoding="utf-8").splitlines(),
        [plant_root, PROJECT_ROOT],
        path.name,
        class_map=class_map,
        variant_prefix=variant_prefix,
    )


def _resolve_split_lines(
    lines: list[str],
    bases: list[Path],
    source_name: str,
    class_map: dict[str, str] | None = None,
    variant_prefix: str = "",
) -> list[tuple[str, str]]:
    """Resolve raw split lines to (class_name, absolute_path) pairs.

    Paths resolve against ``bases`` in order (or absolutely). ``class_map``
    (from the manifest sidecar) overrides folder-name class derivation for
    field layouts where folders are farm/plot ids, not classes.
    ``variant_prefix`` (e.g. "color/") is stripped when looking up class_map —
    manifest rel paths are relative to the image-tree root.
    """
    pairs: list[tuple[str, str]] = []
    missing = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        candidate = Path(line)
        abs_path = (
            candidate
            if candidate.is_absolute() and candidate.exists()
            else _first_existing(*(b / line for b in bases))
        )
        if abs_path is not None:
            manifest_rel = (
                line[len(variant_prefix):] if variant_prefix and line.startswith(variant_prefix) else line
            )
            cls = (class_map or {}).get(manifest_rel) or Path(line).parent.name
            pairs.append((cls, str(abs_path)))
        else:
            missing += 1
    if missing:
        log.warning("%s: %d listed images not found on disk", source_name, missing)
    return pairs


def _first_existing(*candidates: Path) -> Path | None:
    for c in candidates:
        if c.exists():
            return c
    return None


def digest_of(text: str) -> int:
    """Process-independent 31-bit digest of a string."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


class CropDiseaseDataset(Dataset):
    """PlantVillage dataset returning image + target + deterministic metadata."""

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        image_size: int = 192,
        train: bool = False,
        subset_per_class: int | None = None,
    ) -> None:
        if subset_per_class is not None:
            by_class: dict[str, list[tuple[str, str]]] = {}
            for pair in pairs:
                by_class.setdefault(pair[0], []).append(pair)
            pairs = []
            for cls in sorted(by_class):
                pairs.extend(by_class[cls][:subset_per_class])
            log.info("Subset to %d per class → %d samples", subset_per_class, len(pairs))

        self.samples = pairs
        self.image_size = image_size
        self.transform = train_transform(image_size) if train else eval_transform(image_size)

        counts = Counter(cls for cls, _ in self.samples)
        self.class_counts = counts
        if counts:
            total = sum(counts.values())
            n_cls = len(counts)
            self.weights = [total / (n_cls * counts[cls]) for cls, _ in self.samples]
        else:
            self.weights = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        cls, path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        target = class_to_index(cls)
        self.class_to_target = None  # populated lazily via class_to_target()

        # Deterministic context synthesis from the file path digest.
        digest = digest_of(path)
        crop_id = TAXONOMY.crop_id(cls)
        stage_idx = digest % len(GROWTH_STAGES)
        region_idx = digest % len(REGIONS)
        month = digest % 12 + 1
        region = REGIONS[region_idx]
        weather = synthesize_weather(region, month, path)

        return {
            "image": self.transform(img),
            "target": torch.tensor(target, dtype=torch.long),
            "crop_id": torch.tensor(crop_id, dtype=torch.long),
            "stage_idx": torch.tensor(stage_idx, dtype=torch.long),
            "region_idx": torch.tensor(region_idx, dtype=torch.long),
            "weather": weather.to_tensor(),
        }


    def target_for(self, path: str) -> int:
        """Target index for a sample path (test helper)."""
        return class_to_index(next(cls for c, p in self.samples if p == path for cls in [c]))


class CropDiseaseDataModule:
    """Bundles dataset construction + loaders behind the config object."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def loaders(
        self, batch_size: int | None = None, split_manifest: dict[str, list[str]] | None = None
    ) -> dict[str, DataLoader]:
        """DataLoaders. ``split_manifest`` (Phase 2, optional) bypasses split
        files entirely: {"train": [rel_paths], "val": [...]} from
        manifest.leakage_safe_split — used by retraining pipelines that hold
        the manifest in memory."""
        cfg = self.cfg
        bs = batch_size or cfg.training.batch_size
        plant_root, _ = _resolve_plant_root(cfg)
        if split_manifest is not None:
            csv_sidecar = _find_manifest_csv(plant_root)
            class_map = class_map_from_sidecar(csv_sidecar) if csv_sidecar else None
            bases = [plant_root, plant_root / VARIANT_DIRS[cfg.data.variant], PROJECT_ROOT]
            train_pairs = _resolve_split_lines(
                split_manifest["train"], bases, "split_manifest[train]", class_map=class_map
            )
            val_pairs = _resolve_split_lines(
                split_manifest["val"], bases, "split_manifest[val]", class_map=class_map
            )
        else:
            train_f, val_f = resolve_split_paths(cfg)
            train_pairs = load_split_lines(train_f)
            val_pairs = load_split_lines(val_f)
        train_ds = CropDiseaseDataset(
            train_pairs,
            image_size=cfg.data.image_size,
            train=True,
            subset_per_class=cfg.data.train_subset_per_class,
        )
        val_ds = CropDiseaseDataset(
            val_pairs,
            image_size=cfg.data.image_size,
            train=False,
            subset_per_class=cfg.data.val_subset_per_class,
        )
        common = dict(num_workers=cfg.data.workers, pin_memory=torch.cuda.is_available())
        return {
            "train": DataLoader(train_ds, batch_size=bs, shuffle=True, **common),
            "val": DataLoader(val_ds, batch_size=bs, shuffle=False, **common),
            "test": DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0),
        }
