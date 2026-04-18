import cv2
import numpy as np
import pytest
from pill_counter import _fallback_ref_contour


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
