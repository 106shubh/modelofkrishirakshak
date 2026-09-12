"""Training script for cropguard."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import Config, load_config
from ..data.datamodule import CropDiseaseDataModule
from ..hardware import recommended_batch_size, use_amp
from .evaluate import fit_temperature
from ..models.baseline import BaselineClassifier
from ..models.multimodal import MultimodalClassifier
from ..taxonomy import TAXONOMY

log = logging.getLogger(__name__)


def build_model(cfg: Config, num_classes: int, pretrained: bool | None = None) -> nn.Module:
    """Build the configured model variant (baseline vs multimodal)."""
    if pretrained is None:
        pretrained = cfg.model.pretrained
    if cfg.model.fusion == "image_only":
        return BaselineClassifier(
            num_classes=num_classes,
            backbone=cfg.model.backbone,
            pretrained=pretrained,
            dropout=cfg.model.dropout,
        )
    return MultimodalClassifier(
        num_classes=num_classes,
        backbone=cfg.model.backbone,
        pretrained=pretrained,
        dropout=cfg.model.dropout,
        fusion=cfg.model.fusion,
        context_dim=cfg.model.context_dim,
    )


def build_model_from_checkpoint(ckpt: dict, device: torch.device, cfg: Config) -> nn.Module:
    """Build + load a model from a checkpoint dict (shared by all consumers).

    The checkpoint records the architecture it was trained with (ckpt["cfg"]) —
    weight shapes must match those keys, not the serving/training process's
    config. Only model-building keys are honored; serving knobs (weather,
    thresholds, severity bands) stay under the caller's control.
    """
    ckpt_cfg = Config.from_dict(ckpt["cfg"]) if "cfg" in ckpt else cfg
    model = build_model(ckpt_cfg, num_classes=len(TAXONOMY.classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _novelty_reference(cfg: Config, model: nn.Module, device: torch.device, split: str = "val") -> torch.Tensor:
    """Embedding reference set for the novelty detector (known classes only)."""
    from ..inference.novelty import collect_embeddings

    datamodule = CropDiseaseDataModule(cfg)
    loaders = datamodule.loaders(batch_size=cfg.training.batch_size)
    embeddings, _ = collect_embeddings(model, loaders[split], device)
    return embeddings


def train(
    cfg: Config,
    output_dir: Path,
    resume_from: Path | None = None,
) -> None:
    """Train the model.

    Args:
        cfg: Configuration
        output_dir: Directory to save checkpoints and logs
        resume_from: Optional checkpoint to resume from
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Setup data
    output_dir.mkdir(parents=True, exist_ok=True)
    datamodule = CropDiseaseDataModule(cfg)
    loaders = datamodule.loaders(batch_size=recommended_batch_size(cfg.training.batch_size, cfg.data.image_size, cfg.model.backbone))

    # Setup model
    model = build_model(cfg, num_classes=len(TAXONOMY.classes)).to(device)

    # Loss (§1: label smoothing — 0 keeps the old behavior)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.eval.label_smoothing)

    # §3.2 two-stage fine-tuning. Optimizer is REBUILT at each stage boundary —
    # requires_grad changes and per-group LRs both demand fresh groups.
    #   stage "frozen":  backbone.requires_grad = False (fusion + head only)
    #   stage "last":    last backbone block unfrozen at lr * 0.25, head at lr
    #   stage "full":    everything trainable; backbone at lr * unfreeze_lr_factor
    freeze_epochs = min(cfg.training.freeze_epochs, cfg.training.epochs)

    def _stage_of(epoch: int) -> str:
        if epoch < freeze_epochs:
            return "frozen"
        if epoch < freeze_epochs + max(1, cfg.training.epochs // 3):
            return "last"
        return "full"

    def _optimizer_for(stage: str) -> torch.optim.AdamW:
        lr = cfg.training.lr
        if stage == "frozen":
            model.freeze_backbone()
            # Head/fusion params only — the backbone is grad-less now.
            trainable = [p for p in model.parameters() if p.requires_grad]
            return torch.optim.AdamW(trainable, lr=lr, weight_decay=cfg.training.weight_decay)
        if stage == "last":
            model.unfreeze_backbone()
            model.freeze_backbone()
            model.unfreeze_last_block()
            head = [p for p in model.parameters() if p.requires_grad and not _in_backbone(model, p)]
            return torch.optim.AdamW(
                [{"params": head, "lr": lr}]
                + model.backbone_param_groups(lr, 0.25),
                weight_decay=cfg.training.weight_decay,
            )
        model.unfreeze_backbone()
        return torch.optim.AdamW(
            [{"params": [p for p in model.parameters() if not _in_backbone(model, p)], "lr": lr}]
            + model.backbone_param_groups(lr, cfg.training.unfreeze_lr_factor),
            weight_decay=cfg.training.weight_decay,
        )

    def _in_backbone(m: nn.Module, p: torch.Tensor) -> bool:
        return any(p is bp for bp in m.backbone.parameters())

    stage = _stage_of(start_epoch) if resume_from and resume_from.exists() else _stage_of(0)
    optimizer = _optimizer_for(stage)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs
    )
    log.info("Stage schedule: freeze_epochs=%d → %s", freeze_epochs, "frozen→last→full")

    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_acc = 0.0
    if resume_from and resume_from.exists():
        log.info(f"Resuming from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_acc", -1.0)
        log.info(f"Resumed from epoch {start_epoch}")

    # AMP / scheduler / early stopping
    use_amp_flag = use_amp(cfg.training.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp_flag and device.type == "cuda")
    total_steps = 0
    if cfg.training.max_steps:
        cfg.training.epochs = max(1, cfg.training.max_steps // max(len(loaders["train"]), 1) + 1)
    patience_left = cfg.training.early_stopping_patience
    # -1.0 (not 0.0): epoch 1 must always define a best, even at 0% val acc —
    # otherwise best_model.pth is never written and every consumer breaks.
    best_val_acc = -1.0

    # Training loop
    for epoch in range(start_epoch, cfg.training.epochs):
        # §3.2 stage transitions: rebuild optimizer when the schedule moves.
        new_stage = _stage_of(epoch)
        if new_stage != stage:
            stage = new_stage
            optimizer = _optimizer_for(stage)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.training.epochs
            )
            log.info("Stage → %s (epoch %d): optimizer rebuilt", stage, epoch + 1)

        # Train phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, batch in enumerate(loaders["train"]):
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            context = {
                "crop_id": batch["crop_id"].to(device),
                "stage_idx": batch["stage_idx"].to(device),
                "region_idx": batch["region_idx"].to(device),
                "weather": batch["weather"].to(device),
            }

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp_flag and device.type == "cuda"):
                outputs = model(images, **context)
                loss = criterion(outputs["logits"], targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = outputs["logits"].max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            total_steps += 1

            if cfg.training.max_steps and total_steps >= cfg.training.max_steps:
                log.info("Reached max_steps=%d — stopping", cfg.training.max_steps)
                break

            if batch_idx % cfg.training.log_every == 0:
                log.info(
                    f"Epoch {epoch+1}/{cfg.training.epochs} "
                    f"Batch {batch_idx}/{len(loaders['train'])} "
                    f"Loss: {loss.item():.4f}"
                )

        scheduler.step()

        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / len(loaders["train"])

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in loaders["val"]:
                images = batch["image"].to(device)
                targets = batch["target"].to(device)
                context = {
                    "crop_id": batch["crop_id"].to(device),
                    "stage_idx": batch["stage_idx"].to(device),
                    "region_idx": batch["region_idx"].to(device),
                    "weather": batch["weather"].to(device),
                }

                outputs = model(images, **context)
                loss = criterion(outputs["logits"], targets)

                val_loss += loss.item()
                _, predicted = outputs["logits"].max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = 100.0 * val_correct / val_total
        avg_val_loss = val_loss / len(loaders["val"])

        log.info(
            f"Epoch {epoch+1}/{cfg.training.epochs} "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        # Early stopping + checkpointing
        if val_acc <= best_val_acc:
            patience_left -= 1
        else:
            patience_left = cfg.training.early_stopping_patience

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            # max() so the saved value includes THIS epoch's improvement — a plain
            # snapshot lags one epoch behind inside best_model.pth.
            "best_val_acc": max(best_val_acc, val_acc),
            "cfg": cfg.to_dict(),
        }
        torch.save(
            checkpoint,
            output_dir / f"checkpoint_epoch_{epoch+1}.pth",
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                checkpoint,
                output_dir / "best_model.pth",
            )
            log.info(f"New best model saved with val_acc: {val_acc:.2f}%")

        if patience_left <= 0:
            log.info(
                "Early stopping (no val improvement for %d epochs)",
                cfg.training.early_stopping_patience,
            )
            break

    # MVP-6: fit calibration temperature on the configured split and stamp it
    # into every checkpoint in out_dir, so the service picks it up wherever it
    # loads from. NFR5: prefer a field-condition split when one exists.
    calib_split = cfg.eval.calibration_split
    if cfg.eval.temperature is None:
        try:
            temperature = fit_temperature(cfg, model, device, split=calib_split)
        except Exception as e:
            log.warning("Temperature fitting failed (T=1.0 default): %s", e)
            temperature = 1.0
    else:
        temperature = cfg.eval.temperature

    # §2: fit the novelty threshold on the same held-out split (known classes).
    novelty_threshold = None
    if cfg.eval.novelty.enabled:
        try:
            from ..inference.novelty import fit_novelty_threshold

            novelty_threshold = fit_novelty_threshold(cfg, model, device, split=calib_split)
        except Exception as e:
            log.warning("Novelty threshold fitting failed (novelty disabled): %s", e)
            novelty_threshold = None

    for ckpt_path in output_dir.glob("*.pth"):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "calibration_temperature" not in ckpt:
            ckpt["calibration_temperature"] = temperature
        if novelty_threshold is not None and "novelty_threshold" not in ckpt:
            ckpt["novelty_threshold"] = novelty_threshold
            ckpt["novelty_reference"] = _novelty_reference(cfg, model, device, split=calib_split)
        torch.save(ckpt, ckpt_path)
    log.info(
        "Calibration T=%.4f%s stamped into checkpoints in %s",
        temperature,
        f", novelty_threshold={novelty_threshold:.4f}" if novelty_threshold is not None else "",
        output_dir,
    )

    log.info(f"Training completed. Best val accuracy: {best_val_acc:.2f}%")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train cropguard model")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./checkpoints"),
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        dest="overrides",
        help="Config overrides, e.g. --set training.epochs=2 data.image_size=160",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load config
    cfg = load_config(args.config, overrides=args.overrides) if args.config else load_config(None, overrides=args.overrides)

    # Output directory: CLI flag wins, else config value, else default
    output_dir = args.output_dir if args.output_dir != Path("./checkpoints") else cfg.training.out_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Start training
    train(cfg, output_dir, args.resume_from)


if __name__ == "__main__":
    main()