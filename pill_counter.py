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
# Mask-quality scorer  (for strategy selection)
# ─────────────────────────────────────────────────────────────────────────────

def _mask_score(mask: np.ndarray, ref_area: float) -> float:
    """
    Lower is better.  Measures how well the contours in *mask* match ref_area.

    Scoring components
    ------------------
    • Coefficient of variation of blob-area ratios (low = uniform blobs).
    • Penalty when a single huge blob dominates (likely the background).
    • Hard rejection for nearly empty/full masks.
    """
    # Reject masks that are nearly empty or nearly full — both are degenerate.
    h, w = mask.shape[:2]
    fill = float(np.sum(mask > 0)) / (h * w + 1e-6)
    if fill < 0.002 or fill > 0.95:
        return 1e9

    kernel = np.ones((3, 3), np.uint8)
    clean  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    clean  = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas  = [cv2.contourArea(c) for c in cnts if cv2.contourArea(c) > ref_area * 0.05]
    if not areas:
        return 1e9

    ratios = [a / ref_area for a in areas]
    mean   = float(np.mean(ratios))

    if mean < 0.08:
        return 1e9

    # When there is only ONE large blob it is almost certainly the background
    # (or the image border), not a pill cluster.  std=0 in this case yields a
    # deceptively perfect score of 0.0 — penalise it explicitly.
    if len(areas) == 1 and mean > 1.5:
        return mean          # large penalty proportional to blob size

    # Exclude "merged cluster" outlier blobs from the CV calculation.
    # A blob > 3× ref_area (or > 6× median) is a merged group of pills;
    # including it in the CV inflates variance and unfairly penalises correct
    # pill masks (e.g., a white mask with one 5× cluster + 26 single-pill blobs).
    median_ratio   = float(np.median(ratios))
    outlier_thresh = max(3.0, median_ratio * 6.0)
    regular        = [r for r in ratios if r <= outlier_thresh]

    if not regular:
        return 1e9

    base = float(np.std(regular) / (float(np.mean(regular)) + 1e-6))

    # Penalise regular blobs > 8×  (background contamination within range).
    large_penalty = sum(max(0.0, r - 8.0) * 0.25 for r in regular)

    # Heavy penalty for very large outlier blobs (> 10× ref_area).
    # These cannot plausibly be merged pills; they are background regions
    # (e.g., the whole tray captured by otsu_inv).  This prevents degenerate
    # masks from winning purely because their 1–2 regular blobs have low CV.
    outlier_penalty = sum(max(0.0, r - 10.0) * 0.50 for r in ratios
                          if r > outlier_thresh)

    # Reward masks with many individual-pill-sized blobs (more segmentation
    # evidence). n = count of regular blobs only — no credit for outliers.
    n = max(1.0, float(len(regular)))
    return (base + large_penalty + outlier_penalty) / np.sqrt(n)


# ─────────────────────────────────────────────────────────────────────────────
# Mask builders
# ─────────────────────────────────────────────────────────────────────────────

def _detect_surface_roi(enhanced_bgr: np.ndarray) -> np.ndarray | None:
    """
    Detect the primary colored surface (e.g., a medication tray) and return a
    binary mask of its interior.  Returns None if no dominant surface is found.

    Works by finding the largest chromatic (saturated) region in the image.
    Pills are typically white/achromatic, so the colored background surface
    can be used to define the region of interest that contains the pills.
    """
    h, w  = enhanced_bgr.shape[:2]

    # Work on a downscaled version for speed; ROI mask is upscaled at the end
    scale  = 0.25
    small  = cv2.resize(enhanced_bgr, (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_AREA)
    hsv_s  = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    sat_s  = hsv_s[:, :, 1]
    val_s  = hsv_s[:, :, 2]

    # Chromatic mask: reasonably saturated and not very dark
    chromatic = np.uint8(((sat_s > 55) & (val_s > 50)) * 255)

    # Close gaps — pills sitting on the surface create holes in the chromatic mask
    k_close = np.ones((15, 15), np.uint8)
    filled  = cv2.morphologyEx(chromatic, cv2.MORPH_CLOSE, k_close, iterations=3)
    k_open  = np.ones((5,  5),  np.uint8)
    filled  = cv2.morphologyEx(filled,   cv2.MORPH_OPEN,  k_open,  iterations=2)

    cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    largest      = max(cnts, key=cv2.contourArea)
    sh, sw       = filled.shape
    largest_area = cv2.contourArea(largest)

    # Only use ROI if the surface is at least 10 % of the (small) image
    if largest_area < sh * sw * 0.10:
        return None

    # Fill the detected surface contour at small scale, then upscale
    small_roi = np.zeros((sh, sw), dtype=np.uint8)
    cv2.drawContours(small_roi, [largest], -1, 255, cv2.FILLED)
    roi = cv2.resize(small_roi, (w, h), interpolation=cv2.INTER_NEAREST)

    return roi


def _white_pill_mask(enhanced_bgr: np.ndarray, mean_ref_value: float) -> np.ndarray:
    """
    Binary mask for white/light-coloured pills.

    Uses brightness thresholding AND an achromatic (low-saturation) filter so
    that colored backgrounds (blue trays, colored surfaces) are excluded even
    when their brightness overlaps with the pills.
    """
    gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV)

    # Achromatic pixels: low saturation → white/light pill, NOT a colored surface.
    # Threshold of 55 excludes teal/light-blue backgrounds (saturation ≈ 60–180)
    # while keeping white or off-white pills (saturation ≈ 5–40).
    achromatic = np.uint8((hsv[:, :, 1] < 55) * 255)

    if mean_ref_value > 127:
        _, otsu  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        hard_val = max(int(mean_ref_value * 0.72), 95)
        _, hard  = cv2.threshold(gray, hard_val, 255, cv2.THRESH_BINARY)
        bright   = cv2.bitwise_and(otsu, hard)
        # Restrict to achromatic regions: removes colored background blobs
        return cv2.bitwise_and(bright, achromatic)
    else:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return mask


def _color_pill_mask(enhanced_bgr: np.ndarray, color_profiles: list) -> np.ndarray:
    """
    Combined binary mask from all colour profiles.
    Achromatic clusters → V-channel only; chromatic → full HSV range.
    Slight tolerance expansion (×1.15) helps when group-photo lighting
    differs from the reference shot.
    """
    hsv      = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV)
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)

    EXPANSION = 1.15     # widen colour tolerance for group-photo variability

    for p in color_profiles:
        lo = p["hsv_lower"].copy()
        hi = p["hsv_upper"].copy()

        # Expand saturation and value tolerances slightly
        lo[1] = max(0.0,   lo[1] / EXPANSION)
        lo[2] = max(0.0,   lo[2] / EXPANSION)
        hi[1] = min(255.0, hi[1] * EXPANSION)
        hi[2] = min(255.0, hi[2] * EXPANSION)

        if p["is_achromatic"]:
            v     = hsv[:, :, 2]
            s     = hsv[:, :, 1]
            # Enforce low saturation so that colored surfaces (blue trays, etc.)
            # are excluded even when their brightness overlaps the pill range.
            sat_cap = np.uint8(min(55.0, float(hi[1])))
            cm    = np.where(
                (v >= np.uint8(lo[2])) & (v <= np.uint8(hi[2])) & (s <= sat_cap),
                np.uint8(255), np.uint8(0)
            )
        else:
            if p["wraps_hue"]:
                m1 = cv2.inRange(hsv,
                    np.array([lo[0], lo[1], lo[2]], dtype=np.uint8),
                    np.array([179,   hi[1], hi[2]], dtype=np.uint8))
                m2 = cv2.inRange(hsv,
                    np.array([0,     lo[1], lo[2]], dtype=np.uint8),
                    np.array([hi[0], hi[1], hi[2]], dtype=np.uint8))
                cm = cv2.bitwise_or(m1, m2)
            else:
                cm = cv2.inRange(hsv,
                    np.array([lo[0], lo[1], lo[2]], dtype=np.uint8),
                    np.array([hi[0], hi[1], hi[2]], dtype=np.uint8))

        combined = cv2.bitwise_or(combined, cm)

    return combined


def _edge_based_mask(enhanced_bgr: np.ndarray) -> np.ndarray:
    """
    Fallback: Canny edge map → close gaps → fill enclosed regions.
    Works reasonably for pills with strong edge contrast regardless of colour.
    """
    gray  = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 25, 80)
    kernel = np.ones((5, 5), np.uint8)
    # Close small gaps in the edges so each pill outline is a closed contour
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    dilated = cv2.dilate(closed, kernel, iterations=1)

    # Flood-fill background from the image border to isolate enclosed regions
    filled = dilated.copy()
    h, w   = filled.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(filled, flood_mask, (0, 0), 255)
    filled_inv = cv2.bitwise_not(filled)
    result = cv2.bitwise_or(dilated, filled_inv)

    # Sanity check: if the result covers nearly the whole image the edge
    # contours were open (background leaked through).  Fall back to the
    # filled closed-edge regions only.
    fill = np.sum(result > 0) / (h * w + 1e-6)
    if fill > 0.90:
        # Try just filling each found closed contour individually
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fallback = np.zeros_like(closed)
        cv2.drawContours(fallback, cnts, -1, 255, cv2.FILLED)
        return fallback

    return result


def _reference_bg_mask(image_bgr: np.ndarray, bg_model: dict) -> np.ndarray:
    """
    Segment foreground (pills) using the reference background model.

    Projects each pixel's LAB deviation onto the learned background→pill
    direction from the reference image.  Pixels that lie far from the
    background in that direction are pills; background-like pixels score
    near zero.

    IMPORTANT: pass the original (pre-CLAHE) image, not the enhanced one.
    CLAHE equalises local contrast and can eliminate the very brightness
    or colour signal that distinguishes the pill from the background.

    A median-L anchor compensates for overall brightness differences
    between the reference and group photos (different lighting / exposure).
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    bg_mean   = bg_model["bg_mean_lab"].copy().astype(np.float32)   # (3,)
    bg_std    = np.maximum(bg_model["bg_std_lab"].astype(np.float32), 8.0)
    pill_mean = bg_model["pill_mean_lab"].astype(np.float32)         # (3,)

    # Re-anchor L to account for overall brightness shift between images.
    # Most pixels in the group image are background, so median(L) ≈ background L.
    image_median_L = float(np.median(lab[:, :, 0]))
    bg_mean[0] += image_median_L - float(bg_mean[0])

    # Normalised pill direction (background → pill in whitened LAB space)
    direction = (pill_mean - bg_mean) / bg_std          # (3,)

    # Per-pixel normalised deviation from adjusted background
    diff      = lab - bg_mean[np.newaxis, np.newaxis, :]
    norm_diff = diff / bg_std[np.newaxis, np.newaxis, :]

    # Scalar projection onto pill direction
    proj     = np.sum(norm_diff * direction, axis=2)    # (H, W)
    ref_proj = float(np.sum(direction ** 2))            # projection of the pill itself

    # Mark pixels that are ≥ 30 % as different as the reference pill.
    # Floor of 0.3 prevents an almost-empty mask when contrast is very low.
    threshold = max(ref_proj * 0.30, 0.3)
    return np.uint8(proj > threshold) * 255


def _best_mask(enhanced_bgr: np.ndarray,
               color_profiles: list,
               is_white: bool,
               ref_area: float,
               mean_ref_value: float,
               extra_candidates: dict | None = None) -> np.ndarray:
    """
    Build several candidate masks and return the one whose region-size
    distribution best matches ref_area (lowest _mask_score).

    extra_candidates : optional pre-computed masks (e.g. reference-calibrated
                       background subtraction) added before the scored comparison.
    """
    candidates = {}

    # Inject any pre-computed masks first (e.g. ref_bg computed from raw image).
    if extra_candidates:
        candidates.update(extra_candidates)

    if is_white:
        candidates["white"] = _white_pill_mask(enhanced_bgr, mean_ref_value)
    else:
        candidates["color"] = _color_pill_mask(enhanced_bgr, color_profiles)
        candidates["white"] = _white_pill_mask(enhanced_bgr, mean_ref_value)

    candidates["edge"] = _edge_based_mask(enhanced_bgr)

    # Also try Otsu on greyscale directly
    gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    _, otsu     = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY     + cv2.THRESH_OTSU)
    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    candidates["otsu"]     = otsu
    candidates["otsu_inv"] = otsu_inv

    scores = {k: _mask_score(v, ref_area) for k, v in candidates.items()}
    best_key = min(scores, key=scores.__getitem__)

    mask = candidates[best_key]

    # Sanity checks
    h, w = mask.shape
    fill = np.sum(mask > 0) / (h * w)
    if fill < 0.005:                      # almost empty → try runner-up
        scores.pop(best_key)
        if scores:
            mask = candidates[min(scores, key=scores.__getitem__)]
            fill = np.sum(mask > 0) / (h * w)
    if fill > 0.92:                        # almost full → invert
        mask = cv2.bitwise_not(mask)

    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Scale estimation (handles different camera distances for ref vs group photo)
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_scale(mask: np.ndarray, ref_area: float) -> float:
    """
    Estimate the true pill area in *mask* when the group photo was taken
    from a different distance than the reference shot.

    Strategy
    --------
    1. Collect all blobs in the range [5 % – 200 %] of ref_area.
       These are blobs that could plausibly be 1–2 individual pills.
    2. Return the median of the *upper* half of those blobs.
       Larger blobs within the valid range are more likely to be complete,
       isolated pills (smaller blobs may be clipped at boundaries or split).

    Falls back to ref_area when not enough data is available.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return ref_area

    # Collect blobs in [5 %, 200 %] of ref_area — individual-pill candidates
    lo_cap = ref_area * 0.05
    hi_cap = ref_area * 2.0
    candidates = sorted(
        [cv2.contourArea(c) for c in cnts
         if lo_cap <= cv2.contourArea(c) <= hi_cap]
    )

    if not candidates:
        # No blobs in the individual-pill range.
        # All blobs are large → divide the smallest one by its implied n.
        all_areas = sorted([cv2.contourArea(c) for c in cnts if cv2.contourArea(c) >= lo_cap])
        if not all_areas:
            return ref_area
        smallest_large = all_areas[0]
        n_est = max(1, round(smallest_large / ref_area))
        estimated = smallest_large / n_est
        return float(np.clip(estimated, ref_area / 8.0, ref_area * 8.0))

    # Use an area-weighted median rather than the plain or upper-half median.
    # Plain median is pulled down by small noise blobs (white pill case:
    # many tiny background blobs reduce eff_area by half, mis-classifying the
    # pill as a 2-pill region). Upper-half median is biased upward by any
    # merged 2-pill blobs (orange pill case: merged 58K blob pulls estimate to
    # 44K, so the same 58K blob rounds to area_count=1 instead of 2).
    # Weighting each blob by its own area prioritises larger, more pill-like
    # blobs and naturally handles both cases.
    sorted_cands = sorted(candidates)
    total_weight = sum(sorted_cands)
    cumulative   = 0.0
    estimated    = float(sorted_cands[-1])  # fallback: largest blob
    for a in sorted_cands:
        cumulative += a
        if cumulative >= total_weight * 0.5:
            estimated = float(a)
            break

    # Sanity: clamp to 1/8 – 4× the reference area
    estimated = float(np.clip(estimated, ref_area / 8.0, ref_area * 4.0))
    return estimated


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
                     ref_shape: dict | None) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Run watershed on *mask* and count pills in each region.
    Returns (total_count, annotated_image, dist_transform_normalised).
    """
    h, w = mask.shape

    # ── Distance transform ──────────────────────────────────────────────────
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_norm = dist.copy()
    cv2.normalize(dist, dist_norm, 0, 1.0, cv2.NORM_MINMAX)

    # Adaptive foreground threshold:
    # Larger pills relative to image size → higher threshold (less risk of
    # splitting a single pill into two segments).
    pill_fraction = ref_radius / (0.5 * (h + w) + 1e-6)
    fg_thresh = float(np.clip(0.28 + pill_fraction * 1.5, 0.22, 0.52))

    _, sure_fg = cv2.threshold(dist_norm, fg_thresh, 1, cv2.THRESH_BINARY)
    sure_fg    = np.uint8(sure_fg * 255)

    kernel     = np.ones((3, 3), np.uint8)
    sure_bg    = cv2.dilate(mask, kernel, iterations=4)
    unknown    = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers    = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(annotated.copy(), markers)
    annotated[markers == -1] = [0, 0, 200]   # thin red boundary

    # ── Analyse each region ─────────────────────────────────────────────────
    total_count  = 0
    pill_regions = []

    for label in np.unique(markers):
        if label <= 1:
            continue

        region_mask = np.uint8(markers == label) * 255
        cnts, _     = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        contour     = max(cnts, key=cv2.contourArea)
        region_area = cv2.contourArea(contour)

        if region_area < ref_area * 0.10:
            continue

        # ── Estimate count for this region ──────────────────────────────────
        area_count   = max(1, round(region_area / ref_area))
        maxima_count = _count_maxima(dist_norm, region_mask, ref_radius)

        pill_count = _reconcile_counts(area_count, maxima_count, region_area, ref_area)

        # ── Shape validation ─────────────────────────────────────────────────
        if ref_shape is not None and pill_count == 1:
            metrics = _shape_metrics(contour)
            tols    = {"circularity": 0.35,
                       "aspect_ratio": 0.32,
                       "solidity":     0.25}
            if not all(abs(metrics[k] - ref_shape[k]) <= tols[k] for k in tols):
                continue   # reject — shape doesn't match reference pill

        if ref_shape is not None and pill_count > 1:
            # Multi-pill blobs: require reasonable solidity so that highly
            # irregular background blobs (low solidity) are rejected.
            metrics  = _shape_metrics(contour)
            if metrics["solidity"] < ref_shape["solidity"] - 0.45:
                continue   # too irregular to be a pill cluster

        total_count += pill_count
        pill_regions.append((contour, region_area, pill_count))

    return total_count, annotated, dist_norm, pill_regions


def _reconcile_counts(area_count: int, maxima_count: int,
                      region_area: float, ref_area: float) -> int:
    """
    Combine the area-ratio estimate and the local-maxima estimate.

    Strategy
    --------
    • If they agree within 1 → use area_count (lower variance).
    • If maxima >> area  → pills are denser than ref; trust maxima.
    • If area >> maxima  → pills may be stacked; prefer area but cap it.
    • Tie-break: pick whichever gives a ratio closest to an integer.
    """
    if abs(area_count - maxima_count) <= 1:
        return area_count

    # How far each estimate is from a "clean" integer multiple of ref_area
    ratio = region_area / ref_area
    area_err   = abs(ratio - round(ratio))
    maxima_err = abs(ratio - maxima_count)

    if area_err <= maxima_err:
        return area_count
    else:
        return maxima_count


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


def count_pills(group_image_np, ref_area, color_profiles,
                is_white=False, ref_shape=None, bg_model=None):
    """
    Count pills in the group image.

    Parameters
    ----------
    group_image_np : np.ndarray  BGR image containing multiple pills
    ref_area       : float       pixel area of one pill (from analyze_reference)
    color_profiles : list        colour cluster descriptions (from analyze_reference)
    is_white       : bool        True when pill is achromatic & brighter than bg
    ref_shape      : dict | None shape metrics of one pill (for validation)
    bg_model       : dict | None background model from analyze_reference; when
                                 provided it enables reference-calibrated masking
                                 which is far more accurate for subtle pill/bg contrast

    Returns
    -------
    count     : int
    annotated : np.ndarray  BGR image with detection overlay
    """
    if group_image_np is None or group_image_np.size == 0:
        raise ValueError("Group image is empty or invalid.")

    group_image_np = _normalise_channels(group_image_np)
    enhanced       = _enhance(group_image_np)
    annotated      = group_image_np.copy()

    ref_radius     = float(np.sqrt(ref_area / np.pi))
    mean_ref_value = max(p["mean_value"] for p in color_profiles)

    # ── Detect surface ROI (e.g., a colored medication tray) ────────────────
    # Any prominent colored surface in the image is likely the background
    # surface on which the pills rest.  Restricting detection to within this
    # surface eliminates false-positive blobs from the external background
    # (e.g., white marble countertops, table surfaces, etc.).
    surface_roi = _detect_surface_roi(enhanced)

    # ── Reference-calibrated background subtraction (pre-CLAHE) ─────────────
    # Apply the background model against the ORIGINAL image (before CLAHE).
    # CLAHE equates local brightness and destroys the brightness/colour signal
    # that separates subtle pills (e.g. orange on gray) from the background.
    extra_candidates: dict = {}
    if bg_model is not None:
        extra_candidates["ref_bg"] = _reference_bg_mask(group_image_np, bg_model)

    # ── Best mask ────────────────────────────────────────────────────────────
    mask = _best_mask(enhanced, color_profiles, is_white, ref_area, mean_ref_value,
                      extra_candidates=extra_candidates)

    # Apply the surface ROI if one was reliably detected
    if surface_roi is not None:
        h_m, w_m = mask.shape[:2]
        roi_fill = float(np.sum(surface_roi > 0)) / (h_m * w_m)
        if 0.08 <= roi_fill <= 0.92:
            mask = cv2.bitwise_and(mask, surface_roi)

    # ── Auto-scale: estimate true pill size in the group image ───────────────
    # The group photo may be taken from a different distance than the reference
    # shot.  Re-estimate the effective pill area from isolated blobs in the mask
    # so that watershed thresholds and area-ratio counting use the right scale.
    eff_area   = _estimate_scale(mask, ref_area)
    eff_radius = float(np.sqrt(eff_area / np.pi))

    mask = _clean_mask(mask, eff_radius)

    # ── Watershed counting ───────────────────────────────────────────────────
    total, annotated, dist_norm, regions = _watershed_count(
        mask, annotated, eff_area, eff_radius, ref_shape
    )

    # ── Fallback: plain contour analysis if watershed found nothing ──────────
    if total == 0:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < eff_area * 0.10:
                continue
            a_count = max(1, round(area / eff_area))
            m_count = _count_maxima(dist_norm,
                                    np.uint8(
                                        cv2.drawContours(
                                            np.zeros(mask.shape, np.uint8),
                                            [c], -1, 255, -1
                                        )
                                    ), eff_radius)
            pc = _reconcile_counts(a_count, m_count, area, eff_area)

            if ref_shape is not None and pc == 1:
                metrics = _shape_metrics(c)
                tols    = {"circularity": 0.35, "aspect_ratio": 0.32, "solidity": 0.25}
                if not all(abs(metrics[k] - ref_shape[k]) <= tols[k] for k in tols):
                    continue   # reject non-pill-shaped blobs
            elif ref_shape is not None and pc > 1:
                metrics = _shape_metrics(c)
                if metrics["solidity"] < ref_shape["solidity"] - 0.45:
                    continue

            total += pc
            regions.append((c, area, pc))

    annotated = _annotate(annotated, regions, total)
    return total, annotated
