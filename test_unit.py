import cv2
import numpy as np
import pytest
from pill_counter import _fallback_ref_contour, analyze_reference, _achromatic_mask


def make_pill_on_bg(bg_color=(80, 80, 80), pill_color=(230, 230, 230), size=300):
    """White circle centered in a grey image."""
    img = np.full((size, size, 3), bg_color, dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 5, pill_color, -1)
    return img


def test_fallback_finds_central_contour():
    img = make_pill_on_bg()
    c = _fallback_ref_contour(img)
    h, w = img.shape[:2]
    M = cv2.moments(c)
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    assert abs(cx - w / 2) < w * 0.15
    assert abs(cy - h / 2) < h * 0.15
    area = cv2.contourArea(c)
    assert 0.02 * h * w <= area <= 0.85 * h * w


def test_fallback_raises_on_blank_image():
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    with pytest.raises(ValueError, match="Could not isolate"):
        _fallback_ref_contour(img)


# ── Task 2: analyze_reference ─────────────────────────────────────────────────

def test_analyze_reference_returns_histograms():
    img = make_pill_on_bg(bg_color=(50, 120, 200), pill_color=(220, 220, 220))
    ref_area, pill_hist, bg_hist, is_achromatic, ref_shape = analyze_reference(img)

    assert ref_area > 100
    assert pill_hist.shape == (32, 32)
    assert bg_hist.shape == (32, 32)
    assert isinstance(is_achromatic, bool)
    assert "circularity" in ref_shape
    assert "aspect_ratio" in ref_shape
    assert "solidity" in ref_shape


def test_analyze_reference_detects_achromatic():
    img = make_pill_on_bg(bg_color=(30, 30, 30), pill_color=(240, 240, 240))
    _, _, _, is_achromatic, _ = analyze_reference(img)
    assert is_achromatic is True


def test_analyze_reference_detects_chromatic():
    img = make_pill_on_bg(bg_color=(220, 220, 220), pill_color=(30, 120, 200))
    _, _, _, is_achromatic, _ = analyze_reference(img)
    assert is_achromatic is False


def test_analyze_reference_raises_on_empty():
    with pytest.raises(ValueError):
        analyze_reference(np.zeros((0, 0, 3), dtype=np.uint8))


# ── Task 3: _achromatic_mask ──────────────────────────────────────────────────

def test_achromatic_mask_detects_white_region():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:, 100:] = 240  # right half bright white
    mask = _achromatic_mask(img)
    left_fill  = np.sum(mask[:, :100]  > 0)
    right_fill = np.sum(mask[:, 100:] > 0)
    assert right_fill > left_fill * 2
