import cv2
import numpy as np
import pytest
from pill_counter import (_fallback_ref_contour, analyze_reference, _achromatic_mask,
                          _build_probability_mask, _estimate_scale, _reconcile_counts,
                          _watershed_count)


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


# ── Task 4: _build_probability_mask ──────────────────────────────────────────

def make_group_image(bg_color=(80, 80, 80), pill_color=(30, 150, 200), size=400, n=5):
    """Multiple circles of pill_color on bg_color background."""
    img = np.full((size, size, 3), bg_color, dtype=np.uint8)
    radius = size // 10
    for i in range(n):
        cx = (size // (n + 1)) * (i + 1)
        cy = size // 2
        cv2.circle(img, (cx, cy), radius, pill_color, -1)
    return img, radius


def test_backprojection_highlights_pill_colored_regions():
    pill_color = (30, 150, 200)
    bg_color   = (80, 80, 80)
    ref_img = make_pill_on_bg(bg_color=bg_color, pill_color=pill_color, size=200)
    group_img, r = make_group_image(bg_color=bg_color, pill_color=pill_color)

    _, pill_hist, bg_hist, is_achromatic, _ = analyze_reference(ref_img)
    ref_radius = 200 // 5

    mask = _build_probability_mask(group_img, pill_hist, bg_hist, ref_radius, is_achromatic=False)

    h, w = mask.shape
    fill = np.sum(mask > 0) / (h * w)
    assert 0.05 < fill < 0.90, f"fill={fill:.2f} is out of expected range"


def test_backprojection_uses_achromatic_fallback():
    ref_img = make_pill_on_bg(bg_color=(30, 30, 30), pill_color=(240, 240, 240))
    group_img, _ = make_group_image(bg_color=(30, 30, 30), pill_color=(240, 240, 240))
    _, pill_hist, bg_hist, is_achromatic, _ = analyze_reference(ref_img)

    assert is_achromatic is True
    mask = _build_probability_mask(group_img, pill_hist, bg_hist, 40, is_achromatic=True)
    h, w = mask.shape
    fill = np.sum(mask > 0) / (h * w)
    assert 0.01 < fill < 0.99


# ── Task 5: _estimate_scale and _reconcile_counts ─────────────────────────────

def make_mask_with_blobs(blob_area=5000, n=6, size=800):
    mask = np.zeros((size, size), dtype=np.uint8)
    side = int(np.sqrt(blob_area))
    cols = 3
    for i in range(n):
        r, c = divmod(i, cols)
        x = 50 + c * (side + 50)
        y = 50 + r * (side + 50)
        mask[y:y+side, x:x+side] = 255
    return mask


def test_estimate_scale_returns_median_of_candidates():
    blob_area = 4000
    mask = make_mask_with_blobs(blob_area=blob_area, n=6)
    estimated = _estimate_scale(mask, ref_area=blob_area)
    assert 0.6 * blob_area <= estimated <= 1.4 * blob_area


def test_estimate_scale_fallback_on_empty():
    mask = np.zeros((200, 200), dtype=np.uint8)
    result = _estimate_scale(mask, ref_area=1000.0)
    assert result == 1000.0


def test_reconcile_counts_agrees_within_one():
    assert _reconcile_counts(5, 5, 5000, 1000) == 5
    assert _reconcile_counts(5, 6, 5000, 1000) == 5


def test_reconcile_counts_maxima_sanity_check():
    assert _reconcile_counts(3, 7, 3000, 1000) == 4


def test_reconcile_counts_area_wins_otherwise():
    assert _reconcile_counts(4, 1, 4000, 1000) == 4


# ── Task 6: _watershed_count ──────────────────────────────────────────────────

def make_touching_pills(pill_area=3000, n=3, size=300):
    mask = np.zeros((size, size), dtype=np.uint8)
    annotated = np.full((size, size, 3), 200, dtype=np.uint8)
    radius = int(np.sqrt(pill_area / np.pi))
    spacing = radius * 2 - radius // 3
    start_x = size // 2 - spacing * (n - 1) // 2
    for i in range(n):
        cx = start_x + i * spacing
        cy = size // 2
        cv2.circle(mask,      (cx, cy), radius, 255, -1)
        cv2.circle(annotated, (cx, cy), radius, (180, 180, 180), -1)
    return mask, annotated


def test_watershed_counts_touching_pills():
    n = 3
    ref_area = 3000.0
    ref_radius = float(np.sqrt(ref_area / np.pi))
    mask, annotated = make_touching_pills(pill_area=int(ref_area), n=n)

    total, _, _, regions = _watershed_count(mask, annotated, ref_area, ref_radius, ref_shape=None)
    assert abs(total - n) <= 1, f"Expected ~{n} pills, got {total}"
