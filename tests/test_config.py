"""Tests for the config system (defaults, YAML, env, CLI overrides)."""
from __future__ import annotations

from pathlib import Path

import pytest

from cropguard.config import Config, load_config


def test_defaults():
    cfg = Config()
    assert cfg.data.image_size == 192
    assert cfg.model.backbone == "efficientnet_b0"
    assert cfg.model.fusion == "gated"
    assert 0.0 <= cfg.eval.confidence_threshold <= 1.0


def test_validation_rejects_bad_fusion():
    with pytest.raises(Exception):
        Config.model_validate({"model": {"fusion": "nope"}})


def test_yaml_roundtrip(tmp_path: Path):
    cfg = Config()
    p = tmp_path / "c.yaml"
    p.write_text(cfg.to_yaml(), encoding="utf-8")
    loaded = load_config(p)
    assert loaded.to_dict() == cfg.to_dict()


def test_cli_overrides():
    cfg = load_config(None, overrides=["training.epochs=3", "data.image_size=160"])
    assert cfg.training.epochs == 3
    assert cfg.data.image_size == 160


def test_cli_override_rejects_unknown_key():
    with pytest.raises(ValueError, match="Unknown config key"):
        load_config(None, overrides=["training.nonexistent=1"])


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("CROPGUARD_EVAL__CONFIDENCE_THRESHOLD", "0.5")
    monkeypatch.setenv("CROPGUARD_TRAINING__EPOCHS", "2")
    cfg = load_config()
    assert cfg.eval.confidence_threshold == 0.5
    assert cfg.training.epochs == 2


def test_env_ignores_unknown_keys(monkeypatch):
    monkeypatch.setenv("CROPGUARD_BOGUS__KEY", "x")
    cfg = load_config()
    assert cfg == Config()


def test_missing_config_file():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")
