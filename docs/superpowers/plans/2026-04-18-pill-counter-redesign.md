# Pill Counter Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unreliable multi-mask competition strategy in `pill_counter.py` with histogram backprojection, making pill counting accurate across any pill color, shape, and background.

**Architecture:** `analyze_reference` uses GrabCut to isolate the reference pill and extract a 2D Hue-Saturation histogram. `count_pills` uses that histogram to backproject a probability map onto the group image, thresholds it with Otsu, then runs watershed. Achromatic (white/grey) pills fall back to CLAHE + LAB Otsu since their H-S histogram has no discriminating signal.

**Tech Stack:** Python 3.10+, OpenCV (`cv2`), NumPy, SciPy (for `ndimage.maximum_filter`/`ndimage.label`)

---

## File Map

| File | Change |
|------|--------|
| `pill_counter.py` | Rewrite `analyze_reference`, `count_pills`; add `_fallback_ref_contour`, `_achromatic_mask`, `_build_probability_mask`; update `_estimate_scale`, `_reconcile_counts`, `_watershed_count`; remove `_mask_score`, `_best_mask`, `_white_pill_mask`, `_color_pill_mask`, `_edge_based_mask`, `_reference_bg_mask`, `_detect_surface_roi` |
| `app.py` | Update call sites to use new `analyze_reference` / `count_pills` signatures |
| `test_pill.py` | Update to new API |
| `test_unit.py` | New file — pytest unit tests with synthetic images |

---

## Task 1: Add `_fallback_ref_contour` helper and write its unit test

**Files:**
- Modify: `pill_counter.py`
- Create: `test_unit.py`

- [ ] **Step 1: Write the failing test in `test_unit.py`**

```python
# test_unit.py
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
    # Centroid should be near the image center (within 15% of image size)
    assert abs(cx - w / 2) < w * 0.15
    assert abs(cy - h / 2) < h * 0.15
    # Area should be plausible (2%–85% of image)
    area = cv2.contourArea(c)
    assert 0.02 * h * w <= area <= 0.85 * h * w


def test_fallback_raises_on_blank_image():
    img = np.full((200, 200, 3), 128, dtype=np.uint8)  # uniform — no contours
    with pytest.raises(ValueError, match="Could not isolate"):
        _fallback_ref_contour(img)
```

- [ ] **Step 2: Run the test to verify it fails**

```
cd C:\Users\Sripriya\OneDrive\Documents\GitHub\ISpyPill
python -m pytest test_unit.py -v
```

Expected: `ImportError` or `AttributeError` — `_fallback_ref_contour` does not exist yet.

- [ ] **Step 3: Add `_fallback_ref_contour` to `pill_counter.py`**

Add this function after the `_enhance` function (before `_count_maxima`):

```python
def _fallback_ref_contour(image_np: np.ndarray) -> np.ndarray:
    """
    Otsu-based fallback for reference pill extraction when GrabCut fails.
    Tries both normal and inverted Otsu; picks the contour closest to image
    center that passes area and solidity filters.
    """
    h, w = image_np.shape[:2]
    img_area = h * w
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    center = np.array([w / 2.0, h / 2.0])
    best, best_dist = None, float("inf")

    for flags in [cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU]:
        _, thresh = cv2.threshold(gray, 0, 255, flags)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if not (0.02 * img_area <= area <= 0.85 * img_area):
                continue
            hull_a = cv2.contourArea(cv2.convexHull(c)) + 1e-6
            if area / hull_a < 0.3:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            d = float(np.hypot(cx - center[0], cy - center[1]))
            if d < best_dist:
                best_dist, best = d, c

    if best is None:
        raise ValueError(
            "Could not isolate the reference pill. "
            "Use a photo with the pill clearly visible against a contrasting background."
        )
    return best
```

- [ ] **Step 4: Run the test to verify it passes**

```
python -m pytest test_unit.py::test_fallback_finds_central_contour test_unit.py::test_fallback_raises_on_blank_image -v
```

Expected: Both PASS.

- [ ] **Step 5: Commit**

```bash
git add pill_counter.py test_unit.py
git commit -m "feat: add _fallback_ref_contour helper with unit tests"
```

---

## Task 2: Rewrite `analyze_reference` using GrabCut + HS histograms

**Files:**
- Modify: `pill_counter.py`
- Modify: `test_unit.py`

- [ ] **Step 1: Add the failing test for the new `analyze_reference` signature**

Append to `test_unit.py`:

```python
from pill_counter import analyze_reference


def test_analyze_reference_returns_histograms():
    img = make_pill_on_bg(bg_color=(50, 120, 200), pill_color=(220, 220, 220))
    ref_area, pill_hist, bg_hist, is_achromatic, ref_shape = analyze_reference(img)

    assert ref_area > 100, "ref_area should be non-trivial"
    assert pill_hist.shape == (32, 32), "pill_hist must be 32x32 HS histogram"
    assert bg_hist.shape  == (32, 32), "bg_hist must be 32x32 HS histogram"
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
```

- [ ] **Step 2: Run to verify the tests fail**

```
python -m pytest test_unit.py -k "analyze_reference" -v
```

Expected: Tests fail because `analyze_reference` currently returns `(ref_area, color_profiles, is_white, ref_shape, background_model)`.

- [ ] **Step 3: Replace the `analyze_reference` function in `pill_counter.py`**

Delete the entire existing `analyze_reference` function (lines 721–997) and replace with:

```python
def analyze_reference(image_np):
    """
    Analyse a single reference-pill image using GrabCut for robust isolation.

    Returns
    -------
    ref_area     : float      — pixel area of one pill
    pill_hist    : np.ndarray — 32×32 Hue-Saturation histogram of pill pixels
    bg_hist      : np.ndarray — 32×32 Hue-Saturation histogram of background pixels
    is_achromatic: bool       — True when >80% of pill pixels have saturation < 30
    ref_shape    : dict       — circularity, aspect_ratio, solidity
    """
    if image_np is None or image_np.size == 0:
        raise ValueError("Reference image is empty or invalid.")

    image_np = _normalise_channels(image_np)
    h, w = image_np.shape[:2]
    img_area = h * w

    # ── GrabCut: initialise with central 60% rectangle ─────────────────────
    mx, my = int(w * 0.20), int(h * 0.20)
    rect = (mx, my, w - 2 * mx, h - 2 * my)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    gc_mask = np.zeros((h, w), np.uint8)

    try:
        cv2.grabCut(image_np, gc_mask, rect, bgd_model, fgd_model, 5,
                    cv2.GC_INIT_WITH_RECT)
        gc_binary = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
    except Exception:
        gc_binary = np.zeros((h, w), np.uint8)

    # ── Select contour nearest image centre from GrabCut result ────────────
    center = np.array([w / 2.0, h / 2.0])
    ref_contour = None
    best_dist = float("inf")

    cnts, _ = cv2.findContours(gc_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if not (0.02 * img_area <= area <= 0.85 * img_area):
            continue
        hull_a = cv2.contourArea(cv2.convexHull(c)) + 1e-6
        if area / hull_a < 0.3:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        d = float(np.hypot(cx - center[0], cy - center[1]))
        if d < best_dist:
            best_dist, ref_contour = d, c

    # ── Fallback if GrabCut found nothing valid ─────────────────────────────
    if ref_contour is None:
        ref_contour = _fallback_ref_contour(image_np)

    ref_area = float(cv2.contourArea(ref_contour))
    if ref_area < 80:
        raise ValueError("Reference pill appears too small. Please use a closer photo.")

    ref_shape = _shape_metrics(ref_contour)

    # ── Build pill and background masks ─────────────────────────────────────
    pill_mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(pill_mask, [ref_contour], -1, 255, -1)
    bg_mask = cv2.bitwise_not(pill_mask)

    erode_k = np.ones((5, 5), np.uint8)
    pill_inner = cv2.erode(pill_mask, erode_k, iterations=2)
    bg_inner   = cv2.erode(bg_mask,   erode_k, iterations=3)

    # Use original image HSV — CLAHE compresses saturation and confuses colour
    hsv_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)

    pm = pill_inner if cv2.countNonZero(pill_inner) > 20 else pill_mask
    bm = bg_inner   if cv2.countNonZero(bg_inner)   > 20 else bg_mask

    # ── 2D Hue-Saturation histograms (32 × 32 bins) ─────────────────────────
    pill_hist = cv2.calcHist([hsv_img], [0, 1], pm, [32, 32], [0, 180, 0, 256])
    bg_hist   = cv2.calcHist([hsv_img], [0, 1], bm, [32, 32], [0, 180, 0, 256])
    cv2.normalize(pill_hist, pill_hist, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(bg_hist,   bg_hist,   0, 255, cv2.NORM_MINMAX)

    # ── Achromatic detection ─────────────────────────────────────────────────
    pill_hsv_px = hsv_img[pill_mask == 255]
    if len(pill_hsv_px) > 0:
        is_achromatic = bool(np.mean(pill_hsv_px[:, 1] < 30) > 0.80)
    else:
        is_achromatic = True

    return float(ref_area), pill_hist, bg_hist, is_achromatic, ref_shape
```

- [ ] **Step 4: Run the analyze_reference tests**

```
python -m pytest test_unit.py -k "analyze_reference" -v
```

Expected: All 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add pill_counter.py test_unit.py
git commit -m "feat: rewrite analyze_reference with GrabCut and HS histograms"
```

---

## Task 3: Add `_achromatic_mask` helper and its unit test

**Files:**
- Modify: `pill_counter.py`
- Modify: `test_unit.py`

- [ ] **Step 1: Add the failing test**

Append to `test_unit.py`:

```python
from pill_counter import _achromatic_mask


def test_achromatic_mask_detects_white_region():
    # Dark background with a bright white region on the right half
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:, 100:] = 240  # right half bright white
    mask = _achromatic_mask(img)
    # Right half should have significantly more foreground than left half
    left_fill  = np.sum(mask[:, :100]  > 0)
    right_fill = np.sum(mask[:, 100:] > 0)
    assert right_fill > left_fill * 2
```

- [ ] **Step 2: Run to verify it fails**

```
python -m pytest test_unit.py::test_achromatic_mask_detects_white_region -v
```

Expected: `ImportError` — `_achromatic_mask` does not exist yet.

- [ ] **Step 3: Add `_achromatic_mask` to `pill_counter.py`**

Add this function right after `_enhance` and before `_fallback_ref_contour`:

```python
def _achromatic_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Binary mask for white/grey pills: CLAHE on LAB L-channel then Otsu.
    Automatically picks normal or inverted threshold (whichever covers ≤50%).
    """
    enhanced = _enhance(bgr)
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0]
    _, mask = cv2.threshold(l_ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = mask.shape
    fill = float(np.sum(mask > 0)) / (h * w)
    return cv2.bitwise_not(mask) if fill > 0.5 else mask
```

- [ ] **Step 4: Run the test**

```
python -m pytest test_unit.py::test_achromatic_mask_detects_white_region -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pill_counter.py test_unit.py
git commit -m "feat: add _achromatic_mask helper with unit test"
```

---

## Task 4: Implement `_build_probability_mask` with histogram backprojection

**Files:**
- Modify: `pill_counter.py`
- Modify: `test_unit.py`

- [ ] **Step 1: Add the failing test**

Append to `test_unit.py`:

```python
from pill_counter import _build_probability_mask


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
    pill_color = (30, 150, 200)   # BGR
    bg_color   = (80, 80, 80)
    ref_img = make_pill_on_bg(bg_color=bg_color, pill_color=pill_color, size=200)
    group_img, r = make_group_image(bg_color=bg_color, pill_color=pill_color)

    _, pill_hist, bg_hist, is_achromatic, _ = analyze_reference(ref_img)
    ref_radius = 200 // 5  # approx ref pill radius

    mask = _build_probability_mask(group_img, pill_hist, bg_hist, ref_radius, is_achromatic=False)

    # At least 30% of the image should be detected (5 circles × ~3% each)
    h, w = mask.shape
    fill = np.sum(mask > 0) / (h * w)
    assert 0.05 < fill < 0.90, f"fill={fill:.2f} is out of expected range"


def test_backprojection_uses_achromatic_fallback():
    # Achromatic pill: should route to LAB Otsu, not backprojection
    ref_img = make_pill_on_bg(bg_color=(30, 30, 30), pill_color=(240, 240, 240))
    group_img, _ = make_group_image(bg_color=(30, 30, 30), pill_color=(240, 240, 240))
    _, pill_hist, bg_hist, is_achromatic, _ = analyze_reference(ref_img)

    assert is_achromatic is True
    mask = _build_probability_mask(group_img, pill_hist, bg_hist, 40, is_achromatic=True)
    h, w = mask.shape
    fill = np.sum(mask > 0) / (h * w)
    assert 0.01 < fill < 0.99
```

- [ ] **Step 2: Run to verify it fails**

```
python -m pytest test_unit.py -k "backprojection" -v
```

Expected: `ImportError` — `_build_probability_mask` does not exist yet.

- [ ] **Step 3: Add `_build_probability_mask` to `pill_counter.py`**

Add this function after `_achromatic_mask`:

```python
def _build_probability_mask(group_bgr: np.ndarray,
                             pill_hist: np.ndarray,
                             bg_hist: np.ndarray,
                             ref_radius: float,
                             is_achromatic: bool) -> np.ndarray:
    """
    Return a binary mask of pill pixels in group_bgr.

    Chromatic path  — histogram backprojection on the ratio histogram.
    Achromatic path — CLAHE + LAB L-channel Otsu (brightness-based).
    """
    if is_achromatic:
        return _achromatic_mask(group_bgr)

    # Ratio histogram: bins where pills dominate background get high values
    ratio_hist = (pill_hist + 1e-6) / (bg_hist + 1e-6)
    cv2.normalize(ratio_hist, ratio_hist, 0, 255, cv2.NORM_MINMAX)
    ratio_hist = ratio_hist.astype(np.float32)

    # Backproject: each pixel's (H, S) maps to its "pill likelihood"
    hsv_group = cv2.cvtColor(group_bgr, cv2.COLOR_BGR2HSV)
    prob_map = cv2.calcBackProject(
        [hsv_group], [0, 1], ratio_hist, [0, 180, 0, 256], 1
    )

    # Smooth to connect adjacent pill pixels and suppress isolated noise
    sigma = max(3.0, ref_radius * 0.10)
    ksize = int(sigma * 3) * 2 + 1    # next odd number
    prob_map = cv2.GaussianBlur(prob_map, (ksize, ksize), sigma)

    # Otsu threshold on probability map
    _, mask = cv2.threshold(prob_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Degenerate check: fall back to achromatic mask if coverage is extreme
    h, w = mask.shape
    fill = float(np.sum(mask > 0)) / (h * w)
    if fill < 0.01 or fill > 0.90:
        return _achromatic_mask(group_bgr)

    return mask
```

- [ ] **Step 4: Run the backprojection tests**

```
python -m pytest test_unit.py -k "backprojection" -v
```

Expected: Both PASS.

- [ ] **Step 5: Commit**

```bash
git add pill_counter.py test_unit.py
git commit -m "feat: add _build_probability_mask with histogram backprojection"
```

---

## Task 5: Update `_estimate_scale` and `_reconcile_counts`

**Files:**
- Modify: `pill_counter.py`
- Modify: `test_unit.py`

- [ ] **Step 1: Add failing tests**

Append to `test_unit.py`:

```python
from pill_counter import _estimate_scale, _reconcile_counts


def make_mask_with_blobs(blob_area=5000, n=6, size=800):
    """Mask with n square blobs of approximately blob_area pixels each."""
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
    # Should be within 40% of true blob area
    assert 0.6 * blob_area <= estimated <= 1.4 * blob_area


def test_estimate_scale_fallback_on_empty():
    mask = np.zeros((200, 200), dtype=np.uint8)
    result = _estimate_scale(mask, ref_area=1000.0)
    assert result == 1000.0


def test_reconcile_counts_agrees_within_one():
    assert _reconcile_counts(5, 5, 5000, 1000) == 5
    assert _reconcile_counts(5, 6, 5000, 1000) == 5   # within 1 → use area

def test_reconcile_counts_maxima_sanity_check():
    # maxima >= 2× area → add 1 to area count
    assert _reconcile_counts(3, 7, 3000, 1000) == 4

def test_reconcile_counts_area_wins_otherwise():
    # maxima = 1, area = 4 → area wins (not 2× over)
    assert _reconcile_counts(4, 1, 4000, 1000) == 4
```

- [ ] **Step 2: Run to verify tests fail**

```
python -m pytest test_unit.py -k "estimate_scale or reconcile_counts" -v
```

Expected: Some FAIL because `_estimate_scale` is not yet importable from outside (it's currently only called internally, but it exists — the existing function should be importable. The `_reconcile_counts` tests will likely fail due to the changed logic).

- [ ] **Step 3: Replace `_estimate_scale` in `pill_counter.py`**

Find and replace the entire `_estimate_scale` function with:

```python
def _estimate_scale(mask: np.ndarray, ref_area: float) -> float:
    """
    Estimate true pill area in mask when group photo distance differs from reference.

    Collects blobs in [20%, 300%] of ref_area (single-pill candidates) and returns
    their plain median. Falls back to ref_area when no candidates are found.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return ref_area

    lo_cap = ref_area * 0.20
    hi_cap = ref_area * 3.0
    candidates = sorted(
        [cv2.contourArea(c) for c in cnts if lo_cap <= cv2.contourArea(c) <= hi_cap]
    )

    if not candidates:
        all_areas = sorted(
            [cv2.contourArea(c) for c in cnts if cv2.contourArea(c) >= ref_area * 0.15]
        )
        if not all_areas:
            return ref_area
        smallest = all_areas[0]
        n_est = max(1, round(smallest / ref_area))
        return float(np.clip(smallest / n_est, ref_area / 8.0, ref_area * 8.0))

    estimated = float(np.median(candidates))
    return float(np.clip(estimated, ref_area / 8.0, ref_area * 4.0))
```

- [ ] **Step 4: Replace `_reconcile_counts` in `pill_counter.py`**

Find and replace the entire `_reconcile_counts` function with:

```python
def _reconcile_counts(area_count: int, maxima_count: int,
                      region_area: float, ref_area: float) -> int:
    """
    Primary: use area_count.
    Sanity check: if maxima_count >= 2× area_count, pills are likely over-merged — add 1.
    """
    if abs(area_count - maxima_count) <= 1:
        return area_count
    if maxima_count >= 2 * area_count:
        return area_count + 1
    return area_count
```

- [ ] **Step 5: Run the tests**

```
python -m pytest test_unit.py -k "estimate_scale or reconcile_counts" -v
```

Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add pill_counter.py test_unit.py
git commit -m "feat: simplify _estimate_scale and _reconcile_counts"
```

---

## Task 6: Update `_watershed_count` with per-component foreground threshold

**Files:**
- Modify: `pill_counter.py`
- Modify: `test_unit.py`

- [ ] **Step 1: Add the failing test**

Append to `test_unit.py`:

```python
from pill_counter import _watershed_count


def make_touching_pills(pill_area=3000, n=3, size=300):
    """
    n touching circles in a row — tests that watershed can separate them.
    Returns (mask, annotated_copy).
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    annotated = np.full((size, size, 3), 200, dtype=np.uint8)
    radius = int(np.sqrt(pill_area / np.pi))
    spacing = radius * 2 - radius // 3   # slight overlap
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
    ref_shape = {"circularity": 1.0, "aspect_ratio": 1.0, "solidity": 1.0}

    total, _, _, regions = _watershed_count(mask, annotated, ref_area, ref_radius, ref_shape=None)
    # Should count all 3 pills (allow ±1 tolerance for touching separation)
    assert abs(total - n) <= 1, f"Expected ~{n} pills, got {total}"
```

- [ ] **Step 2: Run to verify test fails or reveals the gap**

```
python -m pytest test_unit.py::test_watershed_counts_touching_pills -v
```

Note the output — the test may already pass or may show the counting is off. Either way, we proceed to update the function.

- [ ] **Step 3: Replace `_watershed_count` in `pill_counter.py`**

Find and replace the entire `_watershed_count` function with:

```python
def _watershed_count(mask: np.ndarray,
                     annotated: np.ndarray,
                     ref_area: float,
                     ref_radius: float,
                     ref_shape: dict | None) -> tuple[int, np.ndarray, np.ndarray, list]:
    """
    Run watershed on mask and count pills in each region.
    Foreground threshold is computed per connected component (0.35 × local max)
    to avoid over- or under-splitting when small and large blobs coexist.
    Returns (total_count, annotated_image, dist_transform_normalised, pill_regions).
    """
    h, w = mask.shape

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_norm = dist.copy()
    cv2.normalize(dist, dist_norm, 0, 1.0, cv2.NORM_MINMAX)

    # Per-component foreground: 0.35 × each component's local maximum
    sure_fg = np.zeros((h, w), dtype=np.uint8)
    num_labels, comp_labels = cv2.connectedComponents(mask)
    for lbl in range(1, num_labels):
        comp_mask = np.uint8(comp_labels == lbl)
        comp_dist = dist_norm * comp_mask
        local_max = float(comp_dist.max())
        if local_max > 0:
            comp_fg = np.uint8(comp_dist >= 0.35 * local_max) * 255
            sure_fg = cv2.bitwise_or(sure_fg, comp_fg)

    kernel  = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(mask, kernel, iterations=4)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(annotated.copy(), markers)
    annotated[markers == -1] = [0, 0, 200]

    total_count  = 0
    pill_regions = []

    for label in np.unique(markers):
        if label <= 1:
            continue
        region_mask = np.uint8(markers == label) * 255
        cnts, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        contour     = max(cnts, key=cv2.contourArea)
        region_area = cv2.contourArea(contour)
        if region_area < ref_area * 0.10:
            continue

        area_count   = max(1, round(region_area / ref_area))
        maxima_count = _count_maxima(dist_norm, region_mask, ref_radius)
        pill_count   = _reconcile_counts(area_count, maxima_count, region_area, ref_area)

        if ref_shape is not None and pill_count == 1:
            metrics = _shape_metrics(contour)
            tols = {"circularity": 0.40, "aspect_ratio": 0.40, "solidity": 0.45}
            if not all(abs(metrics[k] - ref_shape[k]) <= tols[k] for k in tols):
                continue

        if ref_shape is not None and pill_count > 1:
            metrics = _shape_metrics(contour)
            if metrics["solidity"] < ref_shape["solidity"] - 0.45:
                continue

        total_count += pill_count
        pill_regions.append((contour, region_area, pill_count))

    return total_count, annotated, dist_norm, pill_regions
```

- [ ] **Step 4: Run the watershed test**

```
python -m pytest test_unit.py::test_watershed_counts_touching_pills -v
```

Expected: PASS (±1 pill tolerance).

- [ ] **Step 5: Commit**

```bash
git add pill_counter.py test_unit.py
git commit -m "feat: update _watershed_count with per-component foreground threshold"
```

---

## Task 7: Rewrite `count_pills` and remove dead code

**Files:**
- Modify: `pill_counter.py`
- Modify: `test_unit.py`

- [ ] **Step 1: Add the failing integration test**

Append to `test_unit.py`:

```python
from pill_counter import count_pills


def test_count_pills_synthetic():
    """count_pills should find the circles created by make_group_image."""
    pill_color = (30, 150, 200)
    bg_color   = (80, 80, 80)
    n = 5

    ref_img   = make_pill_on_bg(bg_color=bg_color, pill_color=pill_color, size=200)
    group_img, r = make_group_image(bg_color=bg_color, pill_color=pill_color, n=n)

    ref_area, pill_hist, bg_hist, is_achromatic, ref_shape = analyze_reference(ref_img)
    count, annotated = count_pills(group_img, ref_area, pill_hist, bg_hist,
                                   is_achromatic=is_achromatic, ref_shape=ref_shape)

    assert annotated.shape == group_img.shape
    assert abs(count - n) <= 2, f"Expected ~{n} pills, got {count}"
```

- [ ] **Step 2: Run to verify the test fails**

```
python -m pytest test_unit.py::test_count_pills_synthetic -v
```

Expected: `TypeError` because `count_pills` currently takes `(group_image_np, ref_area, color_profiles, is_white, ref_shape, bg_model)`.

- [ ] **Step 3: Replace `count_pills` in `pill_counter.py`**

Delete the entire existing `count_pills` function (starting from `def count_pills(`) and replace with:

```python
def count_pills(group_image_np, ref_area, pill_hist, bg_hist,
                is_achromatic=False, ref_shape=None):
    """
    Count pills in the group image using histogram backprojection.

    Parameters
    ----------
    group_image_np : np.ndarray  BGR image containing multiple pills
    ref_area       : float       pixel area of one pill (from analyze_reference)
    pill_hist      : np.ndarray  32×32 HS histogram of reference pill pixels
    bg_hist        : np.ndarray  32×32 HS histogram of reference background pixels
    is_achromatic  : bool        True when pill has low saturation (white/grey)
    ref_shape      : dict | None shape metrics for single-pill validation

    Returns
    -------
    count     : int
    annotated : np.ndarray  BGR image with detection overlay
    """
    if group_image_np is None or group_image_np.size == 0:
        raise ValueError("Group image is empty or invalid.")

    group_image_np = _normalise_channels(group_image_np)
    annotated  = group_image_np.copy()
    ref_radius = float(np.sqrt(ref_area / np.pi))

    # Build binary pill mask via backprojection (or achromatic fallback)
    mask = _build_probability_mask(
        group_image_np, pill_hist, bg_hist, ref_radius, is_achromatic
    )

    # Estimate effective pill size in case group photo was at different distance
    eff_area   = _estimate_scale(mask, ref_area)
    eff_radius = float(np.sqrt(eff_area / np.pi))

    mask = _clean_mask(mask, eff_radius)

    total, annotated, dist_norm, regions = _watershed_count(
        mask, annotated, eff_area, eff_radius, ref_shape
    )

    # Fallback: plain contour analysis if watershed found nothing
    if total == 0:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < eff_area * 0.10:
                continue
            a_count  = max(1, round(area / eff_area))
            region_m = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(region_m, [c], -1, 255, -1)
            m_count = _count_maxima(dist_norm, region_m, eff_radius)
            pc = _reconcile_counts(a_count, m_count, area, eff_area)

            if ref_shape is not None and pc == 1:
                metrics = _shape_metrics(c)
                tols = {"circularity": 0.40, "aspect_ratio": 0.40, "solidity": 0.45}
                if not all(abs(metrics[k] - ref_shape[k]) <= tols[k] for k in tols):
                    continue
            elif ref_shape is not None and pc > 1:
                metrics = _shape_metrics(c)
                if metrics["solidity"] < ref_shape["solidity"] - 0.45:
                    continue

            total += pc
            regions.append((c, area, pc))

    annotated = _annotate(annotated, regions, total)
    return total, annotated
```

- [ ] **Step 4: Remove dead code from `pill_counter.py`**

Delete the following functions entirely — they are no longer called:
- `_mask_score`
- `_detect_surface_roi`
- `_white_pill_mask`
- `_color_pill_mask`
- `_edge_based_mask`
- `_reference_bg_mask`
- `_best_mask`

Also remove the `from scipy import ndimage` import only if `_count_maxima` no longer uses it — but `_count_maxima` still uses `ndimage.maximum_filter` and `ndimage.label`, so **keep the scipy import**.

- [ ] **Step 5: Run all unit tests**

```
python -m pytest test_unit.py -v
```

Expected: All tests PASS. If any fail, check function signatures match what is imported in the test.

- [ ] **Step 6: Commit**

```bash
git add pill_counter.py test_unit.py
git commit -m "feat: rewrite count_pills with backprojection pipeline; remove dead mask strategies"
```

---

## Task 8: Update `app.py` to use new API

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Update the `/analyze` endpoint in `app.py`**

Find the block starting with `ref_area, color_profiles, is_white, ref_shape, bg_model = analyze_reference(ref_image)` and replace the entire try block:

```python
    try:
        ref_image   = load_image_from_file(ref_file)
        group_image = load_image_from_file(group_file)

        ref_area, pill_hist, bg_hist, is_achromatic, ref_shape = analyze_reference(ref_image)

        count, annotated_image = count_pills(
            group_image, ref_area, pill_hist, bg_hist,
            is_achromatic=is_achromatic, ref_shape=ref_shape,
        )

        return jsonify({
            "count":              count,
            "annotated_image":    encode_image_to_base64(annotated_image),
            "ref_area_px":        round(ref_area, 1),
            "is_white_pill":      is_achromatic,
            "num_color_clusters": 1,
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 422

    except Exception:
        app.logger.error("Unexpected error:\n" + traceback.format_exc())
        return jsonify({"error": "Image processing failed. Please try again with clearer photos."}), 500
```

- [ ] **Step 2: Verify `app.py` imports are unchanged**

The top of `app.py` still imports `from pill_counter import analyze_reference, count_pills` — no change needed there.

- [ ] **Step 3: Start the Flask server to verify it loads without errors**

```
python app.py
```

Expected output (approximately):
```
 * Running on http://0.0.0.0:5000
```

Press `Ctrl+C` to stop. If you see a traceback instead, fix the import or call-site error before continuing.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "fix: update app.py to use new analyze_reference/count_pills API"
```

---

## Task 9: Update `test_pill.py` for new API and run end-to-end tests

**Files:**
- Modify: `test_pill.py`

- [ ] **Step 1: Replace `test_pill.py` with the updated version**

```python
import cv2
import numpy as np
from PIL import Image, ImageOps
import io

from pill_counter import analyze_reference, count_pills


def load_image(path):
    with open(path, "rb") as f:
        data = f.read()
    pil_image = Image.open(io.BytesIO(data))
    pil_image = ImageOps.exif_transpose(pil_image)
    pil_image = pil_image.convert("RGB")
    w, h = pil_image.size
    if max(w, h) > 2000:
        scale = 2000 / max(w, h)
        pil_image = pil_image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


# ── Test set 1: white oval pills on blue tray ────────────────────────────────
print("=== Test set 1: white oval pills on blue tray ===")
img1_ref = load_image("IMG_1241.JPG")
img1_grp = load_image("IMG_1242.JPG")

ref_area, pill_hist, bg_hist, is_achromatic, ref_shape = analyze_reference(img1_ref)
print(f"ref_area: {round(ref_area, 1)} | is_achromatic: {is_achromatic}")

count1, ann1 = count_pills(img1_ref, ref_area, pill_hist, bg_hist,
                            is_achromatic=is_achromatic, ref_shape=ref_shape)
print(f"Single-pill self-test (IMG_1241): {count1}  (expected: 1)")
cv2.imwrite("result_white_single.jpg", ann1)

count2, ann2 = count_pills(img1_grp, ref_area, pill_hist, bg_hist,
                            is_achromatic=is_achromatic, ref_shape=ref_shape)
print(f"Group photo     (IMG_1242): {count2}  (expected: ~25)")
cv2.imwrite("result_white_group.jpg", ann2)


# ── Test set 2: orange round pills on gray tray ──────────────────────────────
print()
print("=== Test set 2: orange round pills on gray tray ===")
img2_ref = load_image("IMG_1243.jpeg")
img2_grp = load_image("IMG_1244.jpeg")

ref_area2, pill_hist2, bg_hist2, is_achromatic2, ref_shape2 = analyze_reference(img2_ref)
print(f"ref_area: {round(ref_area2, 1)} | is_achromatic: {is_achromatic2}")

count3, ann3 = count_pills(img2_ref, ref_area2, pill_hist2, bg_hist2,
                            is_achromatic=is_achromatic2, ref_shape=ref_shape2)
print(f"Single-pill self-test (IMG_1243): {count3}  (expected: 1)")
cv2.imwrite("result_orange_single.jpg", ann3)

count4, ann4 = count_pills(img2_grp, ref_area2, pill_hist2, bg_hist2,
                            is_achromatic=is_achromatic2, ref_shape=ref_shape2)
print(f"Group photo     (IMG_1244): {count4}  (expected: 5)")
cv2.imwrite("result_orange_group.jpg", ann4)

print()
print("Done. Annotated results written to result_*.jpg")
```

- [ ] **Step 2: Run `test_pill.py` with the real images**

```
cd C:\Users\Sripriya\OneDrive\Documents\GitHub\ISpyPill
python test_pill.py
```

Expected output (approximately):
```
=== Test set 1: white oval pills on blue tray ===
ref_area: <number> | is_achromatic: True
Single-pill self-test (IMG_1241): 1  (expected: 1)
Group photo     (IMG_1242): 24  (expected: ~25)

=== Test set 2: orange round pills on gray tray ===
ref_area: <number> | is_achromatic: False
Single-pill self-test (IMG_1243): 1  (expected: 1)
Group photo     (IMG_1244): 5  (expected: 5)
```

**Acceptance criteria:** each count is within ±2 of the expected value.

- [ ] **Step 3: Inspect annotated output images**

Open `result_white_group.jpg` and `result_orange_group.jpg`. Verify:
- Pill regions are highlighted with colored overlays
- Numbers on the overlays add up to the total banner
- No huge background region is being counted as pills

If counting is severely wrong (off by >5), re-read the Debugging Guide at the bottom of this plan before making code changes.

- [ ] **Step 4: Run the full pytest suite to confirm no regressions**

```
python -m pytest test_unit.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add test_pill.py
git commit -m "fix: update test_pill.py to use new analyze_reference/count_pills API"
```

---

## Debugging Guide

Use this section only if `test_pill.py` produces counts that are severely wrong after Task 9.

### Symptom: Over-counting (e.g., 80 pills detected instead of 25)

The backprojection mask is picking up the background. Check:

1. Run `analyze_reference` on the reference image and inspect `is_achromatic`. If `True` on a clearly colored pill, the pill saturation in the image is low — try photographing with better lighting.
2. Add a debug step to `_build_probability_mask` to save the probability map and the mask:
   ```python
   cv2.imwrite("debug_prob_map.jpg", prob_map)
   cv2.imwrite("debug_mask.jpg", mask)
   ```
   If the mask is nearly all white, the ratio histogram has no discriminating power — the pill and background share the same hue/saturation.
3. If pill and background are similar in H-S space (e.g., both are achromatic), force `is_achromatic=True` and retest.

### Symptom: Under-counting (e.g., 2 detected instead of 25)

Watershed is not separating touching pills. Check:

1. Save the binary mask after `_clean_mask`:
   ```python
   cv2.imwrite("debug_clean_mask.jpg", mask)
   ```
   If pills are merging into one large blob, the morphological close kernel is too large. Check that `eff_radius` is close to the actual pill radius.
2. If pills are correctly separated in the mask but watershed still under-counts, the `_estimate_scale` may be returning a value that is too large (treating merged blobs as single pills). Print `eff_area` and compare to the actual pill area in the mask.

### Symptom: Works on white pills but fails on orange/colored pills

The backprojection path is engaging but producing a bad mask. Check:
1. Print `is_achromatic` — if `True` for an orange pill, the saturation threshold (30) is too low for the lighting conditions. Adjust the threshold in `analyze_reference`:
   ```python
   is_achromatic = bool(np.mean(pill_hsv_px[:, 1] < 30) > 0.80)
   #                                               ^^ try 40 or 50
   ```
2. Save `ratio_hist` as an image to see if there is clear signal:
   ```python
   cv2.imwrite("debug_ratio_hist.jpg", ratio_hist)
   ```

---

*Plan complete. Implement tasks in order. Each task is independently committable and testable.*
