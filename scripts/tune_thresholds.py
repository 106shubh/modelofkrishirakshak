"""Evaluate model on validation set and tune confidence thresholds.

Reports:
- Accuracy, Precision, Recall, F1
- Confidence distribution
- Abstention rate vs Margin
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

# Make `src` importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropguard.config import load_config
from cropguard.data.dataset import CropDiseaseDataModule
from cropguard.inference.service import _resolve_device, _load_model, _resolve_checkpoint, _resolve_temperature
from cropguard.inference.engine import temperature_scale
from cropguard.taxonomy import TAXONOMY

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tune")

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = _resolve_device(cfg)
    
    try:
        model = _load_model(cfg, device)
        ckpt_path = _resolve_checkpoint(cfg)
        temperature = _resolve_temperature(cfg, ckpt_path)
    except Exception as e:
        log.error("Failed to load model: %s", e)
        return 1
        
    model.eval()
    
    datamodule = CropDiseaseDataModule(cfg)
    # Using small batch size for progress visibility
    val_loader = datamodule.loaders(batch_size=16)["val"]
    
    all_targets = []
    all_preds = []
    all_confs = []
    all_margins = []
    
    log.info("Running evaluation on validation set (%d batches)...", len(val_loader))
    
    with torch.no_grad():
        for batch in tqdm(val_loader):
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            crop_id = batch["crop_id"].to(device)
            stage_idx = batch["stage_idx"].to(device)
            region_idx = batch["region_idx"].to(device)
            weather = batch["weather"].to(device)
            
            output = model(
                images,
                crop_id=crop_id,
                stage_idx=stage_idx,
                region_idx=region_idx,
                weather=weather
            )
            
            logits = output["logits"]
            probs = torch.softmax(temperature_scale(logits, temperature), dim=-1)
            
            # Sort probabilities to get top-1 and top-2
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            
            top1_probs = sorted_probs[:, 0]
            top2_probs = sorted_probs[:, 1]
            margins = top1_probs - top2_probs
            preds = sorted_indices[:, 0]
            
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_confs.extend(top1_probs.cpu().numpy())
            all_margins.extend(margins.cpu().numpy())
            
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_confs = np.array(all_confs)
    all_margins = np.array(all_margins)
    
    acc = accuracy_score(all_targets, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)
    
    log.info("--- Model Base Metrics ---")
    log.info("Accuracy:  %.4f", acc)
    log.info("Precision: %.4f", p)
    log.info("Recall:    %.4f", r)
    log.info("F1 Score:  %.4f", f1)
    
    log.info("--- Confidence Distribution ---")
    log.info("Mean Conf: %.4f", np.mean(all_confs))
    log.info("P25 Conf:  %.4f", np.percentile(all_confs, 25))
    log.info("P05 Conf:  %.4f", np.percentile(all_confs, 5))
    
    log.info("--- Margin Distribution ---")
    log.info("Mean Margin: %.4f", np.mean(all_margins))
    log.info("P25 Margin:  %.4f", np.percentile(all_margins, 25))
    log.info("P05 Margin:  %.4f", np.percentile(all_margins, 5))
    
    correct = (all_preds == all_targets)
    incorrect = ~correct
    
    if np.any(incorrect):
        log.info("Incorrect preds median conf: %.4f, median margin: %.4f", 
                 np.median(all_confs[incorrect]), np.median(all_margins[incorrect]))
    
    # Test abstention thresholds
    for margin_thresh in [0.05, 0.1, 0.2, 0.3]:
        for conf_thresh in [0.5, 0.6, 0.7, 0.8]:
            abstained = (all_confs < conf_thresh) | (all_margins < margin_thresh)
            abstain_rate = np.mean(abstained)
            
            if abstain_rate == 1.0:
                continue
                
            retained_correct = correct[~abstained]
            retained_acc = np.mean(retained_correct)
            log.info("Thresh [Conf>=%.2f, Margin>=%.2f] -> Abstain: %.1f%%, Retained Acc: %.4f", 
                     conf_thresh, margin_thresh, abstain_rate * 100, retained_acc)

    return 0

if __name__ == "__main__":
    sys.exit(main())
