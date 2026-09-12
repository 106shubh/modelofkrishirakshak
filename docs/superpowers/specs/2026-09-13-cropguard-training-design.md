# CropGuard Training Pipeline — Design Spec

**Date:** 2026-09-13
**Status:** Draft (awaiting review)
**Owner:** training/coder

## 1. Overview & Goals

Bring the CropGuard training pipeline to parity with the five-phase training
specification for SIH26131:

1. SSL pretraining with collapse monitoring
2. Leakage-safe supervised fine-tuning (farm/plot/date splitting, two-stage unfreeze)
3. Uncertainty sampling as active learning (deduplication, curriculum)
4. Continuous retraining with pseudo-label safeguards
5. Monitoring signal-to-action table

The inference service (`src/cropguard/inference/`) is out of scope. Training
artifacts flow back into checkpoints that the existing service already consumes
via `training.out_dir/best_model.pth`.

## 2. Architecture

```
src/cropguard/training/
├── train.py            # Phase 2 orchestrator (extends existing)
├── ssl.py              # Phase 1: DINO pretraining + linear probe
├── uncertainty.py      # Phase 3: active-learning sampler / curriculum
├── retrain.py          # Phase 4: monthly retraining pipeline + safeguards
└── evaluate.py         # Phase 2 eval + Phase 5 metrics (extends)

src/cropguard/data/
├── splits.py           # Extends manifest with farm/plot/date keys
└── manifest.py         # NEW: leakage-safe manifest builder (farm/plot/date)
```

The existing contracts in `src/cropguard/semi.py` are preserved unchanged:
`PSEUDO_LABEL_THRESHOLD`, `TrainingExample`, `pseudo_label_filter()`,
`promotion_gate()`, `RETRAIN_CADENCE_DAYS`. The retraining pipeline calls them
rather than re-implementing their rules.

## 3. Phase 2 — Leakage-safe supervised fine-tuning (first deliverable)

### 3.1 Splitting

`build_manifest()` already reads the official PlantVillage protocol and rejects
near-duplicate leakage across train/test. Extend it to key on **farm/plot ID +
capture date** when the field data is available:

- New `src/cropguard/data/manifest.py` builds the master manifest as
  `(rel_path, class_name, crop_id, farm_id, plot_id, capture_date)`.
- Cross-split leakage prevention operates at the cluster level: any
  `(farm_id, plot_id)` or perceptual-hash cluster that touches >1 split is
  fully assigned to one split (farmer-level grouping).
- When `crop_id`/`farm_id`/`capture_date` columns are absent (PlantVillage has
  none), fall back to the existing perceptual-hash clustering so the change is
  backward-compatible.
- `CropDiseaseDataModule.loaders()` gains a `split_manifest: dict` parameter;
  the val/test loaders use `field_root` for dual-split eval when present.

### 3.2 Two-stage fine-tuning

`train()` gains a config-driven phase schedule (no code-path branching in
`main.py`):

- Stage 1 (`epochs frozen`): backbone frozen, only fusion + head train.
- Stage 2 (`epochs unfreeze`): gradual unfreeze — last block unfrozen at
  `lr * 0.25`, full unfreeze at `lr * 0.1` for the final third.
- `model.freeze_backbone(bool)` on `MultimodalClassifier`/`BaselineClassifier`
  toggles parameter `requires_grad`.

Losses/metrics already wired: `label_smoothing`, `weight_decay`, `CosineAnnealingLR`,
`early_stopping_patience`. These stay as-is.

### 3.3 Train/val gap logging

`train()` logs, per epoch, `train_acc`/`val_acc` and `train_loss`/`val_loss`
to stdout (existing) **and** to a JSONL metrics file at
`output_dir/metrics.jsonl` with the schema:

```
{"epoch", "train_acc", "val_acc", "train_loss", "val_loss",
 "gap_acc", "gap_loss", "stopped_early"}
```

`gap_acc = train_acc - val_acc` is the Phase 5 signal; a >10pp sustained gap
triggers the "field vs lab drift" alert downstream.

## 4. Phase 1 — SSL pretraining

New module `src/cropguard/training/ssl.py` with:

- `DINOTrainer` — student=EfficientNetV2-S, teacher=momentum-updated EMA of the
  student. Loss: DINO cross-entropy with centering + temperature sharpening
  (τₛ=0.07 student, τₜ=0.04 teacher).
- `SSLTransforms` — plant-safe augmentation set: `RandomResizedCrop`,
  `ColorJitter`, `RandomHorizontalFlip`, `GaussianNoise(0.02)`, `RandomAffine`.
  **Excludes** `GaussianBlur`, `Grayscale`, `Solarization` (these collapse
  spectral signatures on plant tissue per the spec).
- `_linear_probe_eval(model, loader)` — frozen-backbone logistic regression
  evaluated every `probe_every=2000` student steps.
- **Collapse detection gate:** monitor the rank of the probe-class covariance
  matrix and probe accuracy. If `probe_acc < 0.15` AND covariance rank <
  `num_classes/2` for two consecutive checks → halt and raise
  `SSCollapseError`. A stuck probe is the collapse signal (per spec).
- Output checkpoint includes `ssl_backbone_state_dict` + a `linear_probe_acc`
  field; `build_model()` consumes it when `model.pretrained == "ssl"`.

## 5. Phase 3 — Uncertainty sampling & curriculum

New `src/cropguard/training/uncertainty.py`:

- `ActiveLearningSampler` — selects the next batch from unlabeled officer images
  using the abstention bandit's uncertainty (`1 - confidence` above the
  adaptive threshold from `AbstentionBandit`).
- Officer-resolved escalations are always gold (via `TrainingExample(quality="gold")`).
- Dedup: perceptual hash + embedding cosine `> 0.98` within a candidate batch →
  collapse to one example. Reuses `perceptual_hash_distance` from
  `data/validation.py` and the model embedding output.
- `CurriculumLoader` — stage order: `lab_dataset → field_dataset →
  escalation_dataset`, advancing stages when the previous reaches a val-loss
  plateau (patience=2). Stage boundaries are config-driven via
  `training.curriculum: [[lab, field, escalation], ...]`.

## 6. Phase 4 — Continuous retraining

New `src/cropguard/training/retrain.py`:

- Entry point `crohgp-guard-retrain --config configs/retrain.yaml`.
- Assembles gold + filtered-pseudo + weak-label sets via `semi.py`
  contracts: `pseudo_label_filter()` keeps pseudo examples above
  `PSEUDO_LABEL_THRESHOLD` (0.85) only.
- `MAX_PSEUDO_RATIO = 0.35` (new constant in `semi.py`) — the pseudo count is
  hard-capped at 35% of the gold count; excess is dropped.
- Trains Phase-2 two-stage on the merged set, writing
  `checkpoints/retrain_<YYYYMMDD>/best_model.pth`.
- **Regression gate:** `promotion_gate(candidate_field_acc,
  production_field_acc, approved_by)` — candidate must not regress field
  accuracy vs production; missing `approved_by` raises (fail-closed).
- **Disagreement rate logging:** for each pseudo-labeled example, record
  `(agreement=argmax==pseudo_label, confidence)`. Aggregate
  `disagreement_rate = mean(1 - confidence on disagreed)` into
  `metrics.jsonl` under the `pseudo_` prefix.

## 7. Phase 5 — Monitoring

A single `src/cropguard/training/monitor.py` owns the signal-to-action table:

| Signal | Source | Threshold | Action |
|--------|--------|-----------|--------|
| train/val gap | `metrics.jsonl` (`gap_acc`) | > 10pp sustained 3 epochs | raise `FieldDriftAlert`, freeze promotion |
| field accuracy drift | `evaluate.py` dual-split | >5% drop vs last | trigger retrain |
| pseudo-label disagreement | `retrain.py` `pseudo_disagreement_rate` | >0.15 | raise `PseudoDisagreementAlert`, pause promotion |
| linear-probe stagnation | `ssl.py` probe acc | <0.15 for 2 checks | halt SSL, `SSCollapseError` |
| leakage detection | `splits.py` cluster size | >1 split per cluster | move cluster, rewrite manifest |

Each row is a function returning a `SignalReport(name, status, severity, action)`.

## 8. Config additions

Add to `config.py` (defaults preserve current behavior):

```python
# TrainingConfig
ssl_pretrain: bool = False
ssl_probe_every: int = 2000
ssl_max_steps: int | None = None
curriculum: list[list[str]] | None = None
pseudo_max_ratio: float = 0.35
freeze_epochs: int = 0
unfreeze_lr_factor: float = 0.1

# DataConfig (already has field_root)
val_ratio: float = 0.20  # now farm/plot-aware when farm_id present
```

## 9. Testing strategy

- `tests/test_ssl_smoke.py` — 50-step DINO run on `cpu_smoke` dataset; asserts
  `linear_probe_acc` is logged and `SSCollapseError` not raised.
- `tests/test_two_stage.py` — asserts backbone params unfrozen after stage 1.
- `tests/test_manifest_leakage.py` — synthetic farm/plot CSV; asserts a cluster
  is fully in one split.
- `tests/test_retrain_gate.py` — asserts `promotion_gate` fail-closed on
  missing `approved_by`; asserts `MAX_PSEUDO_RATIO` truncation.
- `tests/test_monitor.py` — exercises the signal table against fixture
  `metrics.jsonl`.
- All new tests must pass `pytest tests/test_ssl_smoke.py
  tests/test_two_stage.py tests/test_manifest_leakage.py
  tests/test_retrain_gate.py tests/test_monitor.py` under
  `configs/cpu_smoke.yaml`.

## 10. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| SSL collapse wastes GPU days | Collapse gate halts within 2k steps |
| Leakage through farm/plot metadata | Deterministic manifest rewrite; test asserts cluster containment |
| Pseudo-labels reinforce wrong patterns | Stricter 0.85 threshold + 0.35 ratio cap + disagreement alert |
| Two-stage LR bugs regress eval | Reuse existing checkpoint contract; `cpu_smoke` smoke test |
| Staging order | Ship Phase 2 + monitoring first; SSL/retrain behind `ssl_pretrain`/`retrain` flags |

## 11. Implementation order

1. Phase 2 leakage-safe split + two-stage (`splits.py`, `manifest.py`,
   `train.py`).
2. Phase 5 monitoring (`monitor.py`, `metrics.jsonl` logging).
3. Phase 1 SSL (`ssl.py`).
4. Phase 3 uncertainty/curriculum (`uncertainty.py`).
5. Phase 4 continuous retrain (`retrain.py`, `semi.py` constant).

Each phase gated by the existing `pytest` smoke contract under
`configs/cpu_smoke.yaml`.
