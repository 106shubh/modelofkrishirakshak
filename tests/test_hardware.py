"""Tests for hardware detection, transforms, and logging utilities."""
from __future__ import annotations

import json
import logging

import torch
from torchvision import transforms as T

from cropguard.hardware import get_device, get_device_info, recommended_batch_size, use_amp
from cropguard.logging_util import JsonFormatter, log_extra, setup_logging
from cropguard.data.transforms import eval_transform, train_transform


def test_get_device_valid():
    d = get_device()
    assert d.type in {"cpu", "cuda"}


def test_device_info_keys():
    info = get_device_info()
    assert info["cuda_available"] is False or info["cuda_available"] is True
    assert "device" in info and "torch" in info


def test_recommended_batch_size_cpu_keeps_config():
    assert recommended_batch_size(32, 192, "efficientnet_b0") == 32


def test_use_amp_modes():
    assert use_amp("off") is False
    assert use_amp("on") is True
    assert use_amp("auto") == torch.cuda.is_available()


def test_train_and_eval_transforms():
    tr = train_transform(64)
    ev = eval_transform(64)
    assert isinstance(tr, T.Compose) and isinstance(ev, T.Compose)
    assert any(isinstance(t, T.RandomResizedCrop) for t in tr.transforms)
    assert any(isinstance(t, T.CenterCrop) for t in ev.transforms)


def test_json_log_format():
    setup_logging(fmt="json")
    logger = logging.getLogger("cropguard.test.json")
    buf = _capture(logger, JsonFormatter())
    log_extra(logger, logging.INFO, "hello", key="value", count=3)
    payload = json.loads(buf.getvalue())
    assert payload["msg"] == "hello"
    assert payload["fields"] == {"key": "value", "count": 3}


def _capture(logger, formatter):
    import io

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(formatter)
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    return buf
