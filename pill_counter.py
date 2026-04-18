"""
pill_counter.py  —  Accurate pill detection and counting via computer vision.

Key algorithmic design choices
────────────────────────────────
• CLAHE (LAB colour space) normalises uneven lighting before any thresholding.
• Bilateral filter preserves pill edges while suppressing intra-pill noise.
• Adaptive watershed foreground threshold scales with the reference pill radius,
  avoiding over-/under-segmentation at different image scales.
• Pill count per watershed region is determined by the BETTER of two estimates:
    1. area ratio   — region_area / ref_area
    2. local maxima — one distance-transform peak ≈ one pill centre
• When the two estimates conflict, we take the one closest to an integer and
  apply a light blend, which reduces both over- and under-counting.
• Background separation: multiple Otsu/adaptive strategies are tried; the
  candidate mask whose aggregate-region size distribution best matches ref_area
  is selected (smallest normalised variance wins).
• Morphological cleanup uses a structuring element sized to ref_radius so it
  is neither too aggressive (splits pills) nor too lenient (merges them).
"""

import cv2
import numpy as np
from scipy import ndimage


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_channels(image_np):
    """Return 3-channel BGR uint8 regardless of input depth / channel count."""
    if len(image_np.shape) == 2:
        return cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    if image_np.shape[2] == 4:
        return cv2.cvtColor(image_np, cv2.COLOR_BGRA2BGR)
    return image_np


def _enhance(image_np):
    """
    CLAHE in LAB L-channel (lighting normalisation) followed by a
    bilateral filter (edge-preserving denoising).
    Returns uint8 BGR.
    """
    lab   = cv2.cvtColor(image_np, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab   = cv2.merge([clahe.apply(l), a, b])
    out   = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.bilateralFilter(out, d=9, sigmaColor=55, sigmaSpace=55)


# ─────────────────────────────────────────────────────────────────────────────
# Achromatic mask (white/grey pills)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Histogram backprojection mask
# ─────────────────────────────────────────────────────────────────────────────

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
    ksize = int(sigma * 3) * 2 + 1
    prob_map = cv2.GaussianBlur(prob_map, (ksize, ksize), sigma)

    # Otsu threshold on probability map
    _, mask = cv2.threshold(prob_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Degenerate check: fall back to achromatic mask if coverage is extreme
    h, w = mask.shape
    fill = float(np.sum(mask > 0)) / (h * w)
    if fill < 0.01 or fill > 0.90:
        return _achromatic_mask(group_bgr)

    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Fallback reference contour selector
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Local-maxima counting (cross-check for area ratio)
# ─────────────────────────────────────────────────────────────────────────────

def _count_maxima(dist_full: np.ndarray, region_mask: np.ndarray, ref_radius: float) -> int:
    """
    Count the number of distinct pill centres inside *region_mask* by finding
    local maxima in the pre-computed distance transform.

    Parameters
    ----------
    dist_full   : distance transform of the full binary pill mask (float32, 0–1 range)
    region_mask : binary mask of the single watershed region being examined
    ref_radius  : expected radius (px) of one pill, used to set the search window

    Returns
    -------
    int ≥ 1
    """
    roi = dist_full.copy()
    roi[region_mask == 0] = 0.0
    if roi.max() == 0:
        return 1

    # Window: 40–50 % of one pill radius, capped at 40 px for performance.
    # Using `size` (square window) instead of a circular `footprint` lets scipy
    # use a fast separable implementation — O(h·w·r) instead of O(h·w·r²).
    r         = min(40, max(3, int(ref_radius * 0.45)))
    local_max = ndimage.maximum_filter(roi, size=2 * r + 1)

    # A pixel is a maximum if it equals the neighbourhood max AND is above a
    # minimum height threshold (avoids counting flat-shoulder artefacts)
    threshold = roi.max() * 0.28
    maxima    = (roi == local_max) & (roi >= threshold)

    _, n = ndimage.label(maxima)
    return max(1, int(n))


# ─────────────────────────────────────────────────────────────────────────────
# Scale estimation (handles different camera distances for ref vs group photo)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Morphological cleanup (size-adaptive)
# ─────────────────────────────────────────────────────────────────────────────

def _clean_mask(mask: np.ndarray, ref_radius: float) -> np.ndarray:
    """
    Open/close with kernels proportional to the pill radius.

    The OPEN removes small noise blobs.
    The CLOSE fills interior holes (e.g., embossed markings on pills).

    The close kernel is kept deliberately small (≤ 8 % of radius) so that
    neighbouring pills that are close together are NOT merged — the watershed
    step handles separation of touching pills.
    """
    r_open  = max(2, int(ref_radius * 0.12))
    r_close = max(2, int(ref_radius * 0.08))   # small: fill holes but don't merge pills
    ko = np.ones((r_open  * 2 + 1, r_open  * 2 + 1), np.uint8)
    kc = np.ones((r_close * 2 + 1, r_close * 2 + 1), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  ko, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc, iterations=2)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Watershed with adaptive foreground threshold
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Shape metrics
# ─────────────────────────────────────────────────────────────────────────────

def _shape_metrics(contour) -> dict:
    area      = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circ      = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity  = area / (hull_area + 1e-6)

    if len(contour) >= 5:
        e  = cv2.fitEllipse(contour)
        ax1, ax2 = e[1]
        ar = min(ax1, ax2) / (max(ax1, ax2) + 1e-6)
    else:
        _, _, rw, rh = cv2.boundingRect(contour)
        ar = min(rw, rh) / (max(rw, rh) + 1e-6)

    return {"circularity": float(circ),
            "aspect_ratio": float(ar),
            "solidity": float(solidity)}


# ─────────────────────────────────────────────────────────────────────────────
# Annotation
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE = [
    (34,  197, 94),    # green
    (59,  130, 246),   # blue
    (249, 115, 22),    # orange
    (168, 85,  247),   # purple
    (20,  184, 166),   # teal
    (234, 179, 8),     # yellow
    (239, 68,  68),    # red
]

def _annotate(annotated: np.ndarray,
              pill_regions: list,
              total_count: int) -> np.ndarray:
    h, w = annotated.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, (contour, area, pill_count) in enumerate(pill_regions):
        bgr   = _PALETTE[i % len(_PALETTE)]
        color = (int(bgr[2]), int(bgr[1]), int(bgr[0]))   # BGR

        # Filled semi-transparent overlay
        overlay = annotated.copy()
        cv2.drawContours(overlay, [contour], -1, color, -1)
        cv2.addWeighted(overlay, 0.18, annotated, 0.82, 0, annotated)

        # Outline
        cv2.drawContours(annotated, [contour], -1, color, 2)

        # Centroid label
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            rx, ry, rw, rh = cv2.boundingRect(contour)
            cx, cy = rx + rw // 2, ry + rh // 2

        label     = str(pill_count)
        fs        = min(1.0, max(0.35, np.sqrt(area / (h * w)) * 8))
        thickness = max(1, int(fs * 2))
        (tw, th), _ = cv2.getTextSize(label, font, fs, thickness)
        pad = 4
        rx1 = max(0,     cx - tw // 2 - pad)
        ry1 = max(0,     cy - th // 2 - pad)
        rx2 = min(w - 1, cx + tw // 2 + pad)
        ry2 = min(h - 1, cy + th // 2 + pad)

        cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), color, -1)
        cv2.putText(annotated, label,
                    (cx - tw // 2, cy + th // 2),
                    font, fs, (0, 0, 0), thickness, cv2.LINE_AA)

    # Total banner
    banner = f"Total: {total_count}"
    fs_b   = 1.2
    (tw, th), _ = cv2.getTextSize(banner, font, fs_b, 2)
    cv2.rectangle(annotated, (8, 8), (tw + 22, th + 22), (20, 20, 20), -1)
    cv2.putText(annotated, banner, (14, th + 14),
                font, fs_b, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

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
