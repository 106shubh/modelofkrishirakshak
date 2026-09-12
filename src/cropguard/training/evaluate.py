"""Evaluation script for cropguard."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import Config, load_config
from ..data.datamodule import CropDiseaseDataModule
from ..taxonomy import TAXONOMY

# Lazy import of build_model from .train inside evaluate() — train.py imports
# fit_temperature from this module, so a top-level import would be circular.

log = logging.getLogger(__name__)


def _fit_T(logits: torch.Tensor, targets: torch.Tensor, max_iter: int = 100) -> float:
    """Fit the scalar temperature minimizing NLL on held-out logits (Guo et al. 2017)."""
    log_T = torch.nn.Parameter(torch.zeros(1))  # log-space keeps T > 0
    optimizer = torch.optim.LBFGS([log_T], max_iter=max_iter)
    nll = nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = nll(logits / log_T.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_T.exp().item())


def fit_temperature(
    cfg: Config,
    model: nn.Module,
    device: torch.device,
    split: str = "val",
) -> float:
    """Fit calibration temperature on the held-out validation split (MVP-6).

    NFR5 caveat: PlantVillage val logits are lab-condition. Recalibrate on a
    field-condition test set (PlantDoc / field captures) before trusting the
    numbers in an officer-facing demo.
    """
    datamodule = CropDiseaseDataModule(cfg)
    loaders = datamodule.loaders(batch_size=cfg.training.batch_size)
    if split not in loaders:
        raise ValueError(f"Unknown split '{split}'. Available: {sorted(loaders)}")

    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for batch in loaders[split]:
            images = batch["image"].to(device)
            all_targets.append(batch["target"])
            outputs = model(
                images,
                crop_id=batch["crop_id"].to(device),
                stage_idx=batch["stage_idx"].to(device),
                region_idx=batch["region_idx"].to(device),
                weather=batch["weather"].to(device),
            )
            all_logits.append(outputs["logits"].cpu())

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    T = _fit_T(logits, targets)
    # Guardrail: a degenerate split (tiny/uniform/random) can drive the NLL
    # optimum toward T→∞ (flat toward uniform). Anything outside [0.25, 4] is
    # not a usable calibration signal — clamp rather than ship it.
    # ponytail: fixed clamp because there is no field data yet to distinguish
    # "degenerate split" from "genuinely extreme model"; revisit with real data.
    if not 0.25 <= T <= 4.0:
        log.warning("Fitted T=%.4f outside sane range — clamping to the nearest bound", T)
        T = min(4.0, max(0.25, T))
    log.info("Fitted calibration temperature T=%.4f on '%s' split (%d samples)", T, split, len(targets))
    return T


def _metrics_over_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> dict:
    """Accuracy / loss / abstention over one loader, plus a per-crop breakdown.

    §20: per-crop numbers are required, not optional — an aggregate hides a
    minority crop collapsing behind the volume of data-rich crops. ``per_crop``
    keys are crop names via TAXONOMY; ``worst_crop`` is the minimum-accuracy
    crop (the number the training spec gates checkpointing on).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    total = 0
    abstained = 0
    per_crop: dict[int, list[int]] = {}  # crop_id → [correct, total]

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            outputs = model(
                images,
                crop_id=batch["crop_id"].to(device),
                stage_idx=batch["stage_idx"].to(device),
                region_idx=batch["region_idx"].to(device),
                weather=batch["weather"].to(device),
            )
            loss_sum += criterion(outputs["logits"], targets).item()
            max_prob, predicted = outputs["probs"].max(1)
            hits = predicted.eq(targets)
            total += targets.size(0)
            correct += hits.sum().item()
            abstained += (max_prob < threshold).sum().item()
            for cid, ok in zip(batch["crop_id"].tolist(), hits.tolist()):
                acc = per_crop.setdefault(cid, [0, 0])
                acc[0] += ok
                acc[1] += 1

    crop_metrics = {
        TAXONOMY.crop_name(cid): {
            "accuracy": 100.0 * c / max(n, 1),
            "n": n,
        }
        for cid, (c, n) in sorted(per_crop.items())
    }
    worst = min(crop_metrics.items(), key=lambda kv: kv[1]["accuracy"], default=None)
    return {
        "accuracy": 100.0 * correct / max(total, 1),
        "loss": loss_sum / max(len(loader), 1),
        "abstain_rate": abstained / max(total, 1),
        "n": total,
        "per_crop": crop_metrics,
        "worst_crop": None if worst is None else {"crop": worst[0], **worst[1]},
    }


def evaluate(
    cfg: Config,
    checkpoint: Path | None = None,
) -> dict:
    """Evaluate the model, reporting lab-condition and field-condition splits
    SEPARATELY (§1 acceptance criterion), each with a per-crop breakdown
    (§20: minority-crop performance must never hide inside an average).

    Returns:
        dict with lab_* metrics always (plus lab_per_crop / lab_worst_crop),
        field_* metrics + field_per_crop / field_worst_crop + "domain_gap"
        (field − lab accuracy) when data.field_root points at an existing
        dataset directory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    from .train import build_model_from_checkpoint  # deferred: see import note above

    # Setup model + checkpoint
    if checkpoint is None:
        checkpoint = cfg.training.out_dir / "best_model.pth"
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        log.error(f"Checkpoint not found: {checkpoint}")
        return {}
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model_from_checkpoint(ckpt, device, cfg)
    log.info(f"Loaded checkpoint from epoch {ckpt['epoch'] + 1}")

    threshold = cfg.eval.confidence_threshold

    # Lab-condition split (PlantVillage test protocol)
    datamodule = CropDiseaseDataModule(cfg)
    loaders = datamodule.loaders(batch_size=cfg.training.batch_size)
    lab = _metrics_over_loader(model, loaders["test"], device, threshold)
    log.info(
        "Lab-condition — Loss: %(loss).4f, Acc: %(accuracy).2f%%, Abstention: %(abstain_rate).2f%% (n=%(n)d)",
        lab,
    )
    for crop, m in lab["per_crop"].items():
        log.info("  lab %-12s acc=%6.2f%% (n=%d)", crop, m["accuracy"], m["n"])
    results: dict = {
        f"lab_{k}": v for k, v in lab.items() if k not in {"per_crop", "worst_crop"}
    }
    results["lab_per_crop"] = lab["per_crop"]
    results["lab_worst_crop"] = lab["worst_crop"]

    # Field-condition split (PlantDoc-style / curated field images)
    if cfg.data.field_root is not None and Path(cfg.data.field_root).exists():
        field_cfg = cfg.model_copy(deep=True)
        field_cfg.data.root = cfg.data.field_root
        field_loaders = CropDiseaseDataModule(field_cfg).loaders(
            batch_size=cfg.training.batch_size
        )
        field = _metrics_over_loader(model, field_loaders["test"], device, threshold)
        log.info(
            "Field-condition — Loss: %(loss).4f, Acc: %(accuracy).2f%%, Abstention: %(abstain_rate).2f%% (n=%(n)d)",
            field,
        )
        results.update(
            {f"field_{k}": v for k, v in field.items() if k not in {"per_crop", "worst_crop"}}
        )
        results["field_per_crop"] = field["per_crop"]
        results["field_worst_crop"] = field["worst_crop"]
        results["domain_gap"] = field["accuracy"] - lab["accuracy"]
        log.info("Domain gap (field − lab): %+.2f points", results["domain_gap"])
    else:
        log.info(
            "Field-condition split not configured (data.field_root) — "
            "lab numbers only. §1 acceptance criterion needs both."
        )

    return results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate cropguard model")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        dest="overrides",
        help="Config overrides, e.g. --set data.image_size=160",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    cfg = load_config(args.config, overrides=args.overrides) if args.config else load_config(None, overrides=args.overrides)

    evaluate(cfg, args.checkpoint)


if __name__ == "__main__":
    main()