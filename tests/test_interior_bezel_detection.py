"""Smoke tests for tools/interior_bezel_detection.

scipy is an optional research dependency in this repo (not in
requirements.txt), so these tests verify:

1. The module imports without scipy (numpy only)
2. `detect_interior_bezel_lines` returns a graceful error result rather
   than raising ModuleNotFoundError when scipy is missing
3. When scipy IS available, the detector runs on a synthetic mask
   without crashing
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest


def test_module_imports_without_scipy():
    """Importing the module must succeed even in a venv without scipy."""
    mod = importlib.import_module("tools.interior_bezel_detection")
    assert mod is not None
    assert hasattr(mod, "detect_interior_bezel_lines")
    assert hasattr(mod, "InteriorBezelDetection")


def test_detect_returns_graceful_error_when_scipy_missing(monkeypatch):
    """Force the scipy-availability gate to report missing and verify
    the detector returns an InteriorBezelDetection with a useful
    `debug['error']` rather than raising."""
    from tools import interior_bezel_detection as mod

    # Reset the import cache then force it to "missing"
    if hasattr(mod._try_import_scipy_ndimage, "_cached"):
        delattr(mod._try_import_scipy_ndimage, "_cached")
    monkeypatch.setattr(mod, "_try_import_scipy_ndimage", lambda: None)

    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True

    det = mod.detect_interior_bezel_lines(rgb, mask)
    assert det.signal_quality == 0.0
    assert det.cube_center is None
    assert "scipy" in det.debug.get("error", "").lower()


def test_detect_returns_graceful_error_on_shape_mismatch():
    """Shape-mismatch path must return a detection with a clear error."""
    from tools import interior_bezel_detection as mod

    # Only exercise this if scipy IS available — otherwise the scipy
    # gate fires first and obscures the shape check
    if mod._try_import_scipy_ndimage() is None:
        pytest.skip("scipy not installed in this venv")

    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=bool)  # wrong shape

    det = mod.detect_interior_bezel_lines(rgb, mask)
    assert det.signal_quality == 0.0
    assert "shape mismatch" in det.debug.get("error", "").lower()


def test_detect_runs_on_synthetic_mask_when_scipy_available():
    """When scipy IS available, the detector runs end-to-end on a
    synthetic square mask without crashing. We don't assert on
    quality — just that the dataclass shape is well-formed."""
    from tools import interior_bezel_detection as mod

    if mod._try_import_scipy_ndimage() is None:
        pytest.skip("scipy not installed in this venv")

    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    mask = np.zeros((200, 200), dtype=bool)
    mask[40:160, 40:160] = True

    det = mod.detect_interior_bezel_lines(rgb, mask)
    # Either a "eroded mask too small" / "no signal" graceful result,
    # or a real detection with the dataclass fields present.
    assert isinstance(det, mod.InteriorBezelDetection)
    assert isinstance(det.boundary_lines, list)
    assert isinstance(det.boundary_angles, list)
    assert 0.0 <= det.signal_quality <= 1.0
