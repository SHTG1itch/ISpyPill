"""
Unit tests for the pill-counting pipeline (new saturation/hue/value
tray-ROI architecture).  Fast and deterministic — synthetic shapes only.
Run:  pytest test_unit.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import pytest

from pill_counter import (
    analyze_reference, count_pills,
    _isolate_reference_pill, _fallback_ref_contour, _shape_metrics,
    _build_pill_mask, _estimate_pill_area, _count_pills_in_mask,
    _clean_mask, _hist_mean_sat, _circular_mean_hue, _hue_dist,
)
from tests.synth import make_case


# ── helpers ───────────────────────────────────────────────────────────────────

def pill_on_bg(bg=(80, 80, 80), pill=(230, 230, 230), size=300, shape="round"):
    img = np.full((size, size, 3), bg, np.uint8)
    c = (size // 2, size // 2)
    if shape == "round":
        cv2.circle(img, c, size // 5, pill, -1)
    else:
        cv2.ellipse(img, c, (size // 4, size // 7), 25, 0, 360, pill, -1)
    return img


# ── shape metrics ─────────────────────────────────────────────────────────────

def test_shape_metrics_circle():
    img = pill_on_bg(shape="round")
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    c = max(cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
            key=cv2.contourArea)
    m = _shape_metrics(c)
    assert m["circularity"] > 0.85
    assert m["aspect_ratio"] > 0.85
    assert m["solidity"] > 0.9


def test_hue_helpers():
    assert _hue_dist(10, 110) == pytest.approx(80, abs=1)
    assert _hue_dist(2, 178) == pytest.approx(4, abs=1)         # wrap-around
    px = np.array([[10, 200, 200]] * 50, dtype=np.uint8)        # orange-ish
    assert _circular_mean_hue(px) == pytest.approx(10, abs=2)


# ── reference isolation / analysis ─────────────────────────────────────────────

def test_isolate_reference_pill_is_centered():
    img = pill_on_bg(bg=(50, 120, 200), pill=(230, 230, 230), size=400)
    c = _isolate_reference_pill(img)
    M = cv2.moments(c)
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    assert abs(cx - 200) < 60 and abs(cy - 200) < 60
    assert 0.01 * 400 * 400 < cv2.contourArea(c) < 0.6 * 400 * 400


def test_fallback_ref_contour_alias_works():
    img = pill_on_bg(bg=(50, 120, 200))
    assert cv2.contourArea(_fallback_ref_contour(img)) > 100


def test_analyze_reference_returns_full_profile():
    img = pill_on_bg(bg=(50, 120, 200), pill=(220, 220, 220))
    ref_area, ph, bh, achr, rs = analyze_reference(img)
    assert ref_area > 100
    assert ph.shape == (32, 32) and bh.shape == (32, 32)
    assert isinstance(achr, bool)
    for k in ("circularity", "aspect_ratio", "solidity",
              "pill_sat", "bg_sat", "pill_val", "bg_val", "pill_hue", "bg_hue"):
        assert k in rs


def test_analyze_reference_achromatic_detection():
    white = pill_on_bg(bg=(30, 30, 30), pill=(240, 240, 240))
    colored = pill_on_bg(bg=(220, 220, 220), pill=(30, 120, 200))
    assert analyze_reference(white)[3] is True
    assert analyze_reference(colored)[3] is False


def test_analyze_reference_rejects_empty():
    with pytest.raises(ValueError):
        analyze_reference(np.zeros((0, 0, 3), np.uint8))


# ── mask building / scale estimation ───────────────────────────────────────────

def test_build_pill_mask_highlights_pills():
    ref, grp, _ = make_case(1, "orange", "gray", "round", 6)
    _, ph, bh, achr, rs = analyze_reference(ref)
    mask, roi = _build_pill_mask(grp, ph, bh, achr, rs)
    fill = cv2.countNonZero(mask) / mask.size
    assert 0.01 < fill < 0.6


def test_estimate_pill_area_recovers_blob_size():
    mask = np.zeros((900, 900), np.uint8)
    for (x, y) in [(150, 150), (450, 150), (750, 150), (150, 450), (450, 450)]:
        cv2.circle(mask, (x, y), 60, 255, -1)
    blob = np.pi * 60 * 60
    est = _estimate_pill_area(mask, ref_area=blob, ref_shape=None)
    assert 0.7 * blob <= est <= 1.3 * blob


def test_estimate_pill_area_empty_returns_ref():
    assert _estimate_pill_area(np.zeros((200, 200), np.uint8), 1000.0, None) == 1000.0


# ── counting ───────────────────────────────────────────────────────────────────

def test_count_in_mask_counts_separated_blobs():
    mask = np.zeros((900, 900), np.uint8)
    centers = [(150, 150), (450, 150), (750, 150), (150, 450), (450, 450), (750, 450)]
    for (x, y) in centers:
        cv2.circle(mask, (x, y), 55, 255, -1)
    blob = np.pi * 55 * 55
    total, _, regions = _count_pills_in_mask(
        mask, np.zeros((900, 900, 3), np.uint8), blob, 55.0, None)
    assert total == 6


def test_count_pills_single_pill_is_one():
    img = pill_on_bg(bg=(50, 120, 200), pill=(225, 225, 225), size=400)
    ref_area, ph, bh, achr, rs = analyze_reference(img)
    count, ann = count_pills(img, ref_area, ph, bh, is_achromatic=achr, ref_shape=rs)
    assert count == 1
    assert ann.shape == img.shape


def test_count_pills_rejects_empty():
    with pytest.raises(ValueError):
        count_pills(np.zeros((0, 0, 3), np.uint8), 1000.0,
                    np.zeros((32, 32), np.float32), np.zeros((32, 32), np.float32))


# ── end-to-end accuracy across colours / shapes (synthetic, exact GT) ──────────

@pytest.mark.parametrize("pill,tray,shape", [
    ("white", "blue", "oval"),
    ("orange", "gray", "round"),
    ("red", "white", "round"),
    ("yellow", "black", "capsule"),
    ("blue", "white", "oval"),
    ("green", "gray", "round"),
])
def test_count_pills_accuracy_by_type(pill, tray, shape):
    n = 6
    ref, grp, gt = make_case(hash((pill, tray, shape)) % 1000, pill, tray, shape, n)
    ref_area, ph, bh, achr, rs = analyze_reference(ref)
    count, _ = count_pills(grp, ref_area, ph, bh, is_achromatic=achr, ref_shape=rs)
    assert abs(count - gt) <= 1, f"{pill}/{tray}/{shape}: got {count}, expected {gt}"


# ── real in-repo phone photos (catches overcounts the synthetic harness can't) ──

def _load_real(path, max_dim=2000):
    from PIL import Image, ImageOps
    import io
    with open(path, "rb") as f:
        im = Image.open(io.BytesIO(f.read()))
    im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    if max(w, h) > max_dim:
        s = max_dim / max(w, h)
        im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


# (reference, group, ground_truth, tolerance).  Real phone photos where the tray
# does NOT fill the frame (counter/background at the edges) — the exact condition
# under which an over-eager border-pill recovery overcounts.  The upper bound is
# the real guard: the synthetic harness cannot catch this because its tray always
# fills the frame.
_REAL_CASES = [
    ("IMG_1241.JPG",  "IMG_1242.JPG",  25, 3),   # white oval pills on blue tray
    ("IMG_1243.jpeg", "IMG_1244.jpeg",  5, 1),   # orange round pills on gray tray
]


@pytest.mark.parametrize("ref_file,grp_file,gt,tol", _REAL_CASES)
def test_real_photos_no_overcount(ref_file, grp_file, gt, tol):
    here = os.path.dirname(os.path.abspath(__file__))
    rp, gp = os.path.join(here, ref_file), os.path.join(here, grp_file)
    if not (os.path.exists(rp) and os.path.exists(gp)):
        pytest.skip(f"{ref_file}/{grp_file} not present")
    ref, grp = _load_real(rp), _load_real(gp)
    ra, ph, bh, achr, rs = analyze_reference(ref)
    count, _ = count_pills(grp, ra, ph, bh, is_achromatic=achr, ref_shape=rs)
    # Upper bound is the critical assertion: never overcount (pharmacy priority).
    assert count <= gt + tol, f"{grp_file}: OVERCOUNT {count} (truth ~{gt})"
    assert count >= gt - tol, f"{grp_file}: undercount {count} (truth ~{gt})"


# ── border-cropped pills (regression: edge pills must not be dropped) ───────────

@pytest.mark.parametrize("pill,tray,shape", [
    ("white", "blue", "round"),
    ("white", "gray", "oval"),
    ("red", "gray", "round"),
    ("orange", "blue", "capsule"),
])
def test_count_pills_recovers_border_cropped(pill, tray, shape):
    """Pills touching the frame edge are open notches, not enclosed holes, so the
    tray "hole-fill" model used to drop them entirely — a systematic undercount.
    They must now be recovered when the tray wraps them."""
    from tests.stress_probe import make_case_stress
    ref, grp, gt = make_case_stress(7000 + hash((pill, tray, shape)) % 1000,
                                    pill, tray, shape, n=8, touching=False,
                                    border=True)
    ref_area, ph, bh, achr, rs = analyze_reference(ref)
    count, _ = count_pills(grp, ref_area, ph, bh, is_achromatic=achr, ref_shape=rs)
    assert abs(count - gt) <= 1, f"{pill}/{tray}/{shape}: got {count}, expected {gt}"
