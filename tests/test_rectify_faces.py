"""Unit tests for the face-rectification math."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from tools.rectify_faces import (  # noqa: E402
    _perspective_coeffs,
    extract_stickers_from_rectified,
    rectify_face,
)


def _make_solid_face(color, size=300):
    return Image.new("RGB", (size, size), color)


def test_perspective_coeffs_identity():
    """Square→same-sized square should give a near-identity transform
    (output pixel = input pixel)."""
    src = [(0, 0), (300, 0), (300, 300), (0, 300)]
    coeffs = _perspective_coeffs(src, 300)
    # coeffs are 8 values; the transform should map dst (x,y) back to src (x,y)
    # Apply manually to a sample point
    dx, dy = 150, 150
    a, b, c, d, e, f, g, h = coeffs
    w = g * dx + h * dy + 1
    sx = (a * dx + b * dy + c) / w
    sy = (d * dx + e * dy + f) / w
    assert abs(sx - dx) < 0.01
    assert abs(sy - dy) < 0.01


def test_rectify_face_solid_color_preserves():
    """Rectifying a solid-color region produces a solid-color output."""
    img = Image.new("RGB", (1000, 1000), (123, 45, 200))
    quad = [(200, 200), (600, 200), (600, 600), (200, 600)]
    out = rectify_face(img, quad, output_size=300)
    assert out.size == (300, 300)
    # Sample center pixel
    px = out.getpixel((150, 150))
    assert px == (123, 45, 200)


def test_extract_stickers_solid_color_returns_uniform():
    img = _make_solid_face((200, 50, 30))  # red-ish
    stickers = extract_stickers_from_rectified(img)
    assert len(stickers) == 3
    assert all(len(row) == 3 for row in stickers)
    # All 9 should sample the same color
    for row in stickers:
        for s in row:
            assert s.rgb == (200, 50, 30)
            assert s.classified_color == "red"


def test_extract_stickers_returns_row_major_with_center_at_idx_4():
    """Build a face with 9 distinct-colored cells; verify the sample
    positions land at the correct cell centers."""
    img = Image.new("RGB", (300, 300), (0, 0, 0))
    # Paint 9 distinct cells, each 100×100
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
    ]
    for r in range(3):
        for c in range(3):
            color = colors[r * 3 + c]
            for y in range(r * 100 + 20, (r + 1) * 100 - 20):
                for x in range(c * 100 + 20, (c + 1) * 100 - 20):
                    img.putpixel((x, y), color)
    stickers = extract_stickers_from_rectified(img, patch_fraction=0.2)
    flat = [s for row in stickers for s in row]
    for i, s in enumerate(flat):
        assert s.rgb == colors[i], f"position {i}: got {s.rgb}, expected {colors[i]}"


def test_extract_stickers_center_position_invariant():
    """Position (1,1) is the face center regardless of rectification."""
    img = _make_solid_face((100, 100, 100), size=600)
    stickers = extract_stickers_from_rectified(img)
    cx, cy = stickers[1][1].center_xy
    assert 250 <= cx <= 350
    assert 250 <= cy <= 350


def test_rectify_face_skewed_quad():
    """A skewed (non-square) quad still produces a 300x300 output."""
    img = Image.new("RGB", (800, 600), (40, 40, 40))
    # Paint a colored region inside a parallelogram
    for y in range(100, 400):
        for x in range(150 + (y - 100) // 4, 450 + (y - 100) // 4):
            img.putpixel((x, y), (200, 200, 200))
    # A skewed quad that captures the gray region
    quad = [(150, 100), (450, 100), (525, 400), (225, 400)]
    out = rectify_face(img, quad, output_size=300)
    assert out.size == (300, 300)
    # The center should sample the gray region
    px = out.getpixel((150, 150))
    assert px[0] > 150  # rough check that we hit the bright region


def test_extract_stickers_rejects_zero_patch():
    """patch_fraction so small that no pixels get sampled — still returns shape."""
    img = _make_solid_face((50, 50, 50))
    stickers = extract_stickers_from_rectified(img, patch_fraction=0.001)
    assert len(stickers) == 3
    assert all(len(row) == 3 for row in stickers)
