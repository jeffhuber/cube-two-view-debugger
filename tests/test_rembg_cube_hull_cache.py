"""Regression test for the _rembg_cube_hull cache key bug.

Pre-fix: the cache used `id(processing_image)` as the key. Python may
reuse the id() of a garbage-collected object, so when a sweep
processes many pairs, the cache returns STALE hulls for newly-loaded
images that happen to have the same Python id() as a previously-freed
image.

This caused a 14-sticker regression on Set 17 in the aggregate sweep
(vs identical single-pair runs giving 27/27 correct).

Post-fix: cache key is derived from image content (size + 32x32
thumbnail bytes). Different images get different keys; identical
images get cache hits.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

from tools.evaluate_hybrid_pipeline import _image_content_key  # noqa: E402


def test_same_image_same_key():
    """Reloading the same content produces the same cache key, so
    cached results are reused across calls (which is the cache's whole
    point)."""
    img = Image.new("RGB", (100, 100), (128, 64, 200))
    img2 = Image.new("RGB", (100, 100), (128, 64, 200))
    assert _image_content_key(img) == _image_content_key(img2)


def test_different_images_different_keys():
    """The Set 17 regression bug: two DIFFERENT images shouldn't collide
    in the cache even if Python happens to reuse the same id().
    Content-derived keys make collisions effectively impossible."""
    img_red = Image.new("RGB", (100, 100), (255, 0, 0))
    img_blue = Image.new("RGB", (100, 100), (0, 0, 255))
    assert _image_content_key(img_red) != _image_content_key(img_blue)


def test_different_sizes_different_keys():
    """Same color, different size — different cache key. The size is in
    the key so a 100x100 red image and a 200x200 red image don't
    collide."""
    img_a = Image.new("RGB", (100, 100), (128, 128, 128))
    img_b = Image.new("RGB", (200, 100), (128, 128, 128))
    assert _image_content_key(img_a) != _image_content_key(img_b)


def test_id_collisions_no_longer_cause_false_hits():
    """End-to-end regression: simulate a sweep where Python reuses an
    id() across two distinct images. Before the fix, the second image
    would receive the first's cached hull. After the fix, the second
    image gets its own cache key and a fresh computation.

    We can't easily force id() collisions in a test, but we can
    verify that distinct images yield distinct keys deterministically.
    The Set 17 regression manifested as: single-pair run gave 27/27
    correct stickers on side A; full-corpus sweep gave 14/27 because
    the cached hull from an earlier pair was being returned. With
    content-based keys, the bug is structurally impossible — different
    pixel content yields different keys."""
    import random
    random.seed(42)
    images = []
    for _ in range(50):
        # 50 distinct synthetic images
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        size = (
            random.randint(50, 200),
            random.randint(50, 200),
        )
        images.append(Image.new("RGB", size, color))
    keys = {_image_content_key(img) for img in images}
    assert len(keys) == 50, (
        f"50 distinct images produced only {len(keys)} unique cache keys"
    )
