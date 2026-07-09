"""
pill_counter.py  —  Accurate pill detection and counting via computer vision.

Design philosophy
─────────────────
The previous pipeline segmented pills purely by colour across the WHOLE image.
That fails catastrophically whenever the surroundings contain pill-coloured
material (e.g. white pills on a blue tray sitting on a white counter — the
counter gets counted, the pills get missed).

This rewrite is built on a different, far more robust idea:

    "A pill is an object sitting ON the reference surface (tray)."

So we:
  1.  Learn the reference pill's colour AND its background (tray) colour.
  2.  In the group photo, locate the tray by back-projecting the *background*
      histogram and taking the dominant filled region — this is the ROI.
  3.  Pills are simply the parts of the ROI that are NOT tray-coloured
      (the "holes" in the tray).  Colour of the pill is irrelevant to this
      step, so it works for white, orange, red, blue, two-tone — anything.
  4.  The single-pill size is re-estimated from genuinely isolated pills in
      the group photo (the "singletons"), which auto-corrects an inaccurate
      reference area and handles the camera being closer/farther.
  5.  Touching pills are separated with a distance-transform / watershed step
      seeded by interior peaks (one ridge/peak per pill, robust to ovals and
      capsules), reconciled with an area-ratio estimate for dense clusters.

Every public signature is unchanged so app.py keeps working.
"""

import cv2
import numpy as np
from scipy import ndimage


# ─────────────────────────────────────────────────────────────────────────────
# Basic helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_channels(image_np):
    """Return 3-channel BGR uint8 regardless of input depth / channel count."""
    if image_np is None:
        return None
    if len(image_np.shape) == 2:
        return cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    if image_np.shape[2] == 4:
        return cv2.cvtColor(image_np, cv2.COLOR_BGRA2BGR)
    return image_np


def _shape_metrics(contour) -> dict:
    """circularity, aspect_ratio (minor/major), solidity for a contour."""
    area      = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circ      = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity  = area / (hull_area + 1e-6)

    if len(contour) >= 5:
        (_, _), (ax1, ax2), _ = cv2.fitEllipse(contour)
        ar = min(ax1, ax2) / (max(ax1, ax2) + 1e-6)
    else:
        _, _, rw, rh = cv2.boundingRect(contour)
        ar = min(rw, rh) / (max(rw, rh) + 1e-6)

    return {"circularity": float(np.clip(circ, 0, 1)),
            "aspect_ratio": float(np.clip(ar, 0, 1)),
            "solidity": float(np.clip(solidity, 0, 1))}


def _shape_is_pill_like(metrics: dict, ref_shape: dict | None,
                        strict: bool = True) -> bool:
    """
    Decide whether a contour's shape is consistent with a pill.

    When a reference shape is known we compare against it with generous
    tolerances; otherwise we fall back to absolute sanity limits that accept
    rounds, ovals and capsules but reject thin slivers and very concave blobs.
    """
    if ref_shape is None:
        if strict:
            return (metrics["solidity"] >= 0.80 and
                    metrics["aspect_ratio"] >= 0.22 and
                    metrics["circularity"] >= 0.35)
        return metrics["solidity"] >= 0.65 and metrics["aspect_ratio"] >= 0.18

    if strict:
        tol = {"circularity": 0.45, "aspect_ratio": 0.40, "solidity": 0.30}
        ok = all(abs(metrics[k] - ref_shape[k]) <= tol[k] for k in tol)
        # A pill should still be reasonably solid regardless of the reference.
        return ok and metrics["solidity"] >= 0.70
    return metrics["solidity"] >= max(0.55, ref_shape["solidity"] - 0.35)


# ─────────────────────────────────────────────────────────────────────────────
# Reference pill isolation (ensemble of strategies, scored by pill-likeness)
# ─────────────────────────────────────────────────────────────────────────────

def _candidate_masks(bgr: np.ndarray) -> list:
    """
    Produce several binary foreground candidate masks for a single-pill photo
    using complementary cues.  No single cue is reliable for every pill/tray
    combination, so we generate many and let the scorer pick the best contour.
    """
    h, w = bgr.shape[:2]
    masks = []

    # Light denoise + CLAHE for stable thresholds under uneven lighting.
    blur = cv2.GaussianBlur(bgr, (5, 5), 0)
    lab  = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)
    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    s_ch = hsv[:, :, 1]

    # (1) Saturation Otsu — separates a coloured tray from an achromatic pill
    #     (and vice-versa).  Both polarities tried.
    _, s_hi = cv2.threshold(s_ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(s_hi)
    masks.append(cv2.bitwise_not(s_hi))

    # (2) Brightness (L) Otsu — both polarities.
    _, l_hi = cv2.threshold(l_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(l_hi)
    masks.append(cv2.bitwise_not(l_hi))

    # (3) Distance-from-background-colour, modelled from a central ring around
    #     the (assumed centred) pill.  This is the immediate tray surface, so
    #     the pill stands out even if the image corners contain other material.
    labf = lab.astype(np.float32)
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = min(h, w) / 2.0
    for lo, hi in [(0.30, 0.48), (0.55, 0.75)]:
        ring = (rr >= lo * rmax) & (rr <= hi * rmax)
        if np.count_nonzero(ring) < 50:
            continue
        bg_mean = labf[ring].mean(axis=0)
        dist = np.linalg.norm(labf - bg_mean, axis=2)
        dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, dm = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append(dm)

    # (4) Distance-from-CORNER-colour.  The frame corners are the most likely
    #     places to see pure background even when other pills crowd the edges
    #     or the centred pill is large (which breaks the ring model above).
    #     Crucially this segments a TWO-TONE capsule as one region — both of
    #     its colours differ from the background — where any single-channel
    #     Otsu splits it into halves.
    cp = max(6, int(round(min(h, w) * 0.12)))
    corners = np.concatenate([
        labf[:cp, :cp].reshape(-1, 3),  labf[:cp, -cp:].reshape(-1, 3),
        labf[-cp:, :cp].reshape(-1, 3), labf[-cp:, -cp:].reshape(-1, 3)])
    bg_corner = np.median(corners, axis=0)
    dist = np.linalg.norm(labf - bg_corner, axis=2)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, dm = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(dm)

    return masks


def _split_candidates(mask: np.ndarray) -> list:
    """
    Extra candidate contours from erosion-reconstruction splitting.

    When other pills crowd the reference pill, every threshold mask fuses them
    into one non-pill-shaped blob and the true pill never becomes a candidate
    contour.  Eroding by a fraction of the blob half-width severs the necks
    between touching pills; dilating each separated core back (clipped to the
    original mask) reconstructs one candidate region per pill.
    """
    m = ((mask > 0).astype(np.uint8)) * 255
    if not m.any():
        return []
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    r = float(dt.max())
    if r < 5:
        return []
    # Large blobs are processed at reduced resolution: erosion cost grows with
    # the square of the kernel size, and a blob 150 px deep needs 250 px
    # kernels.  Splitting is geometric, so working at "half-width ~= 30 px"
    # scale loses nothing that matters.
    sc = 1.0 if r <= 35 else 35.0 / r
    if sc < 1.0:
        h_, w_ = m.shape
        m_work = cv2.resize(m, (max(1, int(w_ * sc)), max(1, int(h_ * sc))),
                            interpolation=cv2.INTER_NEAREST)
        r_work = r * sc
    else:
        m_work, r_work = m, r
    out = []
    # Progressive erosion: neck widths vary (side-by-side pills have wide
    # necks, tip-to-tip narrow ones), so no single depth separates every
    # arrangement.  Harvest candidates at each depth that yields a split.
    for f in (0.25, 0.4, 0.55, 0.7, 0.85):
        ek = 2 * int(f * r_work) + 1
        if ek < 3:
            continue
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ek, ek))
        core = cv2.erode(m_work, ker)
        num, labels = cv2.connectedComponents(core, 8)
        if num <= 2:                 # nothing separated at this depth
            continue
        for i in range(1, num):
            comp = np.uint8(labels == i) * 255
            region = cv2.bitwise_and(cv2.dilate(comp, ker), m_work)
            cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            if sc < 1.0:
                c = np.round(c.astype(np.float64) / sc).astype(np.int32)
            out.append(c)
    return out


def _isolate_reference_pill(bgr: np.ndarray):
    """
    Return the contour of the single reference pill, chosen as the most
    pill-like candidate across all strategies in :func:`_candidate_masks`.

    Isolation runs on a downscaled copy (candidate generation uses erosion
    kernels proportional to the pill size, which is prohibitively slow at
    full resolution) and the winning contour is scaled back up.  Colour and
    histogram sampling in analyze_reference still happen at full resolution.
    """
    h0, w0 = bgr.shape[:2]
    scale = 640.0 / max(h0, w0)
    if scale < 1.0:
        small = cv2.resize(bgr, (max(1, int(round(w0 * scale))),
                                 max(1, int(round(h0 * scale)))),
                           interpolation=cv2.INTER_AREA)
        c = _isolate_reference_pill_impl(small)
        c = np.round(c.astype(np.float64) / scale).astype(np.int32)
        c[:, :, 0] = np.clip(c[:, :, 0], 0, w0 - 1)
        c[:, :, 1] = np.clip(c[:, :, 1], 0, h0 - 1)
        return c
    return _isolate_reference_pill_impl(bgr)


def _isolate_reference_pill_impl(bgr: np.ndarray):
    h, w = bgr.shape[:2]
    img_area = float(h * w)
    center = np.array([w / 2.0, h / 2.0])
    diag = float(np.hypot(w, h))

    # Morphology kernel scaled to the image so cleanup is resolution-independent.
    k = max(3, int(round(min(h, w) * 0.01))) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    labf = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0),
                        cv2.COLOR_BGR2LAB).astype(np.float32)
    gx = cv2.Sobel(labf[:, :, 0], cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(labf[:, :, 0], cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    # Pass 1 — cheap geometric terms for every candidate contour.
    prelim = []
    for mask in _candidate_masks(bgr):
        m = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        m = cv2.morphologyEx(m,    cv2.MORPH_CLOSE, kernel, iterations=2)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = list(cnts) + _split_candidates(m)
        for c in cnts:
            area = cv2.contourArea(c)
            frac = area / img_area
            if not (0.001 <= frac <= 0.85):
                continue
            metrics = _shape_metrics(c)
            if metrics["solidity"] < 0.80:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            d = float(np.hypot(cx - center[0], cy - center[1]))

            # A reference pill sits inside the frame; background regions and
            # partially-visible neighbours hug the frame edges.  Touching one
            # border is mildly suspicious, touching several means "this is the
            # background / frame corner", not the pill being photographed.
            bx, by, bw2, bh2 = cv2.boundingRect(c)
            touches = int(bx <= 1) + int(by <= 1) + \
                int(bx + bw2 >= w - 2) + int(by + bh2 >= h - 2)
            border_term = 1.0 if touches == 0 else (0.7 if touches == 1 else 0.2)

            center_term = np.exp(-((d / (0.55 * diag)) ** 2))
            shape_term  = max(0.2, min(1.0, metrics["circularity"] * 1.5))
            solid_term  = metrics["solidity"]

            # Prefer a moderate fraction of the frame (a centred pill close-up),
            # but keep small candidates alive: a pill on a large tray can be
            # a small part of the frame.
            if 0.03 <= frac <= 0.45:
                size_term = 1.0
            elif frac > 0.45:           # very large -> probably background blob
                size_term = 0.35
            elif frac >= 0.01:
                size_term = 0.5
            elif frac >= 0.004:
                size_term = 0.3
            else:
                size_term = 0.2

            # Edge alignment: a real pill's outline sits on image edges along
            # its whole length; a region carved out of a uniform surface (an
            # erosion-split artefact of a large tray blob) has a boundary that
            # mostly crosses flat colour.
            pts = c.reshape(-1, 2)
            gv = grad[np.clip(pts[:, 1], 0, h - 1), np.clip(pts[:, 0], 0, w - 1)]
            edge_frac = float(np.mean(gv > 10.0))
            edge_term = float(np.clip(0.15 + edge_frac * 1.1, 0.15, 1.0))

            cheap = (solid_term * shape_term * center_term * size_term *
                     border_term * edge_term)
            prelim.append((cheap, frac, c))

    # Pass 2 — colour terms (expensive) for the strongest candidates only,
    # computed inside each candidate's bounding window, not the whole frame.
    prelim.sort(key=lambda t: -t[0])
    best = None
    best_score = -1.0
    for cheap, frac, c in prelim[:60]:
        if cheap <= best_score:        # colour terms can only lower the score
            break
        bx, by, bw2, bh2 = cv2.boundingRect(c)
        pad = 4 * k
        x0, y0 = max(0, bx - pad), max(0, by - pad)
        x1, y1 = min(w, bx + bw2 + pad), min(h, by + bh2 + pad)
        win = labf[y0:y1, x0:x1]
        cmask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.drawContours(cmask, [c], -1, 255, -1, offset=(-x0, -y0))
        er = cv2.erode(cmask, kernel)
        inner = er if cv2.countNonZero(er) > 30 else cmask
        ring = cv2.dilate(cmask, kernel, iterations=3)
        ring = cv2.bitwise_and(ring, cv2.bitwise_not(cmask))
        if cv2.countNonZero(ring) < 30:
            continue
        inner_med = np.median(win[inner > 0], axis=0)
        ring_med  = np.median(win[ring > 0], axis=0)

        # Contrast between the candidate's interior and the surface ring
        # around it.  A real pill differs sharply from the surface it sits
        # on; glints, shadow wedges and gaps between pills have interiors
        # almost identical to their surroundings.  Medians: the ring may
        # contain slivers of neighbouring pills, but as long as the majority
        # of it is surface, the median ignores them.
        contrast = float(np.linalg.norm(inner_med - ring_med))
        contrast_term = float(np.clip(contrast / 30.0, 0.1, 1.0))

        score = cheap * contrast_term
        if score > best_score:
            best_score, best = score, c

    if best is None:
        raise ValueError(
            "Could not isolate the reference pill. Use a photo with the pill "
            "clearly visible against a contrasting background."
        )
    return best


# Kept as a public name for backward compatibility / tests.
def _fallback_ref_contour(image_np: np.ndarray) -> np.ndarray:
    return _isolate_reference_pill(_normalise_channels(image_np))


# ─────────────────────────────────────────────────────────────────────────────
# Histogram helpers
# ─────────────────────────────────────────────────────────────────────────────

_HBINS, _SBINS = 32, 32


def _hs_hist(hsv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hist = cv2.calcHist([hsv], [0, 1], mask, [_HBINS, _SBINS], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist


def _backproject(hsv: np.ndarray, hist: np.ndarray) -> np.ndarray:
    return cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)


def _circular_mean_hue(hsv_px: np.ndarray) -> float:
    """Saturation-weighted circular mean hue (OpenCV 0-180) of HSV pixels."""
    if len(hsv_px) == 0:
        return 0.0
    ang = hsv_px[:, 0].astype(np.float32) * (np.pi / 90.0)   # 0..2pi
    wt  = hsv_px[:, 1].astype(np.float32) + 1.0              # weight by saturation
    m = np.arctan2(np.sum(np.sin(ang) * wt), np.sum(np.cos(ang) * wt))
    if m < 0:
        m += 2 * np.pi
    return float(m * 90.0 / np.pi)


def _hue_dist(a: float, b: float) -> float:
    """Circular distance between two OpenCV hues (0-180), result in 0-90."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _hist_mean_sat(hist: np.ndarray) -> float:
    """Average saturation (0-255) represented by a 2D Hue-Saturation histogram."""
    col = hist.sum(axis=0)                       # mass per saturation bin
    total = float(col.sum())
    if total <= 0:
        return 0.0
    centers = (np.arange(_SBINS) + 0.5) * (256.0 / _SBINS)
    return float((col * centers).sum() / total)


# ─────────────────────────────────────────────────────────────────────────────
# Public API: reference analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_reference(image_np):
    """
    Analyse a single reference-pill image.

    Returns
    -------
    ref_area      : float      pixel area of one pill
    pill_hist     : np.ndarray 32x32 Hue-Saturation histogram of pill pixels
    bg_hist       : np.ndarray 32x32 Hue-Saturation histogram of background
    is_achromatic : bool       True when the pill is white/grey (low saturation)
    ref_shape     : dict       circularity, aspect_ratio, solidity
    """
    if image_np is None or image_np.size == 0:
        raise ValueError("Reference image is empty or invalid.")

    image_np = _normalise_channels(image_np)
    h, w = image_np.shape[:2]

    ref_contour = _isolate_reference_pill(image_np)
    ref_area = float(cv2.contourArea(ref_contour))
    if ref_area < 80:
        raise ValueError("Reference pill appears too small. Please use a closer photo.")

    ref_shape = _shape_metrics(ref_contour)

    # Masks for pill / background colour sampling.
    pill_mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(pill_mask, [ref_contour], -1, 255, -1)

    radius = float(np.sqrt(ref_area / np.pi))

    # Erode the pill mask so edge/antialias pixels (often tray-coloured) are
    # excluded from the pill histogram.
    er = max(3, int(round(radius * 0.20))) | 1
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (er, er))
    pill_inner = cv2.erode(pill_mask, erode_k, iterations=1)
    pm = pill_inner if cv2.countNonZero(pill_inner) > 30 else pill_mask

    # Background = an ANNULUS of the surface immediately around the pill (the
    # tray it sits on) — NOT the whole frame.  Sampling the whole frame would
    # pull in distant material of the pill's own colour (e.g. a white counter
    # behind a white pill on a blue tray), which would make pills look like
    # background and be erased.  The annulus is the surface the pill rests on.
    ring_in  = max(3, int(round(radius * 0.30))) | 1
    ring_out = max(ring_in + 2, int(round(radius * 1.60))) | 1
    k_in  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_in, ring_in))
    k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_out, ring_out))
    pill_excl = cv2.dilate(pill_mask, k_in)              # pill + thin guard ring
    pill_wide = cv2.dilate(pill_mask, k_out)
    bg_mask = cv2.bitwise_and(pill_wide, cv2.bitwise_not(pill_excl))
    if cv2.countNonZero(bg_mask) < 200:                  # pill nearly fills frame
        bg_mask = cv2.bitwise_not(pill_excl)

    # Decontaminate the annulus: if other pills of the same type crowd the
    # reference pill, they fall inside the annulus and their colour poisons the
    # "tray" statistics (inverting every pill-vs-tray comparison downstream).
    # Pixels that look like the pill are other pills; model the tray from the
    # LEAST pill-like part of the annulus, then keep only pixels closer to
    # that tray model than to the pill.
    labf_full = cv2.cvtColor(image_np, cv2.COLOR_BGR2LAB).astype(np.float32)
    pill_med = np.median(labf_full[pm > 0], axis=0)
    ann_px = labf_full[bg_mask > 0]
    if len(ann_px) >= 200:
        d_pill = np.linalg.norm(ann_px - pill_med, axis=1)
        far = ann_px[d_pill >= np.percentile(d_pill, 60)]
        tray_med = np.median(far, axis=0)
        keep = (np.linalg.norm(ann_px - tray_med, axis=1) <
                np.linalg.norm(ann_px - pill_med, axis=1))
        if keep.sum() >= 100:
            ys, xs = np.nonzero(bg_mask)
            drop = ~keep
            bg_mask = bg_mask.copy()
            bg_mask[ys[drop], xs[drop]] = 0

    # Use the ORIGINAL image HSV (CLAHE distorts saturation/hue relationships).
    hsv_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
    pill_hist = _hs_hist(hsv_img, pm)
    bg_hist   = _hs_hist(hsv_img, bg_mask)

    # Median saturation / value of pill and surrounding tray.  These let the
    # group stage pick the most discriminative channel: saturation (pale pill on
    # a coloured tray), value (white pill on a black tray) or hue (two colours).
    pill_px = hsv_img[pm > 0]
    bg_px   = hsv_img[bg_mask > 0]
    if len(pill_px) and len(bg_px):
        ref_shape["pill_sat"] = float(np.median(pill_px[:, 1]))
        ref_shape["bg_sat"]   = float(np.median(bg_px[:, 1]))
        ref_shape["pill_val"] = float(np.median(pill_px[:, 2]))
        ref_shape["bg_val"]   = float(np.median(bg_px[:, 2]))
        ref_shape["pill_hue"] = _circular_mean_hue(pill_px)
        ref_shape["bg_hue"]   = _circular_mean_hue(bg_px)

    # Achromatic check from the inner half of the pill bounding box.
    rx, ry, rw, rh = cv2.boundingRect(ref_contour)
    sx, sy = rx + rw // 4, ry + rh // 4
    ex, ey = rx + 3 * rw // 4, ry + 3 * rh // 4
    sample = hsv_img[sy:ey, sx:ex].reshape(-1, 3)
    is_achromatic = bool(len(sample) == 0 or np.mean(sample[:, 1] < 40) > 0.75)

    return ref_area, pill_hist, bg_hist, is_achromatic, ref_shape


# ─────────────────────────────────────────────────────────────────────────────
# Group photo: build the pill mask (objects-on-the-tray model)
# ─────────────────────────────────────────────────────────────────────────────

def _largest_filled_regions(binary: np.ndarray, min_frac: float,
                            img_area: float):
    """
    Keep connected components of *binary* covering >= min_frac of the image,
    then fill their interior holes.

    Returns (kept, filled) where *kept* is the dominant tray-coloured region and
    *filled* additionally includes its enclosed holes.  The set difference
    ``filled & ~kept`` is exactly the objects (pills) sitting on the tray.
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    keep = np.zeros_like(binary)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_frac * img_area:
            keep[labels == i] = 255
    if cv2.countNonZero(keep) == 0:
        return keep, keep
    filled = ndimage.binary_fill_holes(keep > 0).astype(np.uint8) * 255
    return keep, filled


def _recover_border_pills(side_is_pill: np.ndarray, roi: np.ndarray,
                          tray_core: np.ndarray, img_area: float,
                          ctx: dict | None = None) -> np.ndarray:
    """
    Recover pills that touch the image border and are therefore lost by the
    "pill = enclosed hole in the tray" model: a pill at the frame edge is an
    OPEN notch, not a hole, so :func:`binary_fill_holes` never recovers it and
    it is dropped — a systematic undercount whenever a pill is partly cropped.

    A border pill is a pill-side component that (a) touches the image edge,
    (b) lies outside the filled ROI, (c) is wrapped by the tray on its
    non-border sides, and (d) is no larger than a small cluster of pills.  The
    tray-surround AND size gates are deliberately strict so same-coloured
    material OUTSIDE the tray (e.g. a white counter behind white pills) — which
    is large and NOT tray-wrapped — can never be admitted.  This keeps the fix
    from ever introducing an overcount; it only restores genuinely cropped
    pills sitting on the tray.

    Returns a mask of the recovered border-pill pixels (may be all-zero).
    """
    h, w = side_is_pill.shape
    # Recovery is only SAFE when the tray fills the frame: then any pill-coloured
    # material at the image edge must be a cropped pill, because there is no
    # off-tray background there to mistake for one.  If the tray does NOT dominate
    # the image-border ring, real background (a counter/table around a central
    # tray) reaches the edges, and recovering border components would grab it —
    # an overcount.  So bail out unless the tray clearly wraps the whole frame.
    border_ring = np.zeros((h, w), bool)
    border_ring[0, :] = border_ring[-1, :] = border_ring[:, 0] = border_ring[:, -1] = True
    if (roi[border_ring] > 0).mean() < 0.55:
        return np.zeros_like(side_is_pill)

    outside = cv2.bitwise_and(side_is_pill, cv2.bitwise_not(roi))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(outside, 8)
    add = np.zeros_like(side_is_pill)
    tray_bool = tray_core > 0
    rk = max(3, int(round(np.sqrt(img_area) * 0.01))) | 1
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rk, rk))
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x  = stats[i, cv2.CC_STAT_LEFT]
        y  = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        touches = (x == 0 or y == 0 or x + bw >= w or y + bh >= h)
        if not touches:
            continue
        # A cropped pill (or a small just-touching cluster) is small; the
        # external counter / off-tray background is large -> reject it outright.
        if area > 0.06 * img_area or area < 0.0006 * img_area:
            continue
        comp = labels == i
        comp_u = (comp.astype(np.uint8)) * 255
        ring = cv2.dilate(comp_u, ring_kernel) > 0
        ring &= ~comp
        # Ignore the image-border portion of the ring — only the inward sides
        # tell us whether the tray wraps the object.
        ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = False
        ring_n = int(ring.sum())
        if ring_n == 0:
            continue
        if (ring & tray_bool).sum() / ring_n < 0.55:
            continue
        # The mask-side wrap test is not enough when the "tray side" of the
        # split covers the whole background (value/flat-field cues): a bottle
        # cap on the counter is then "tray-wrapped" too.  Require the ring to
        # actually LOOK like the learned tray colour, not merely to fall on
        # the tray side of the split.
        if ctx is not None:
            ring_med = np.median(ctx["labf"][ring], axis=0)
            if float(np.linalg.norm(ring_med - ctx["tray_lab"])) > 38.0:
                continue
        add[comp] = 255
    return add


def _tray_and_pills(side_is_tray: np.ndarray, side_is_pill: np.ndarray,
                    img_area: float, ctx: dict | None = None):
    """
    Given a binary split of the image into a tray side and a pill side, return
    (roi, pill_mask): the filled tray region and the pill pixels inside it.

    The tray is the dominant connected region of *side_is_tray*; filling its
    holes recovers the pills resting on it, and intersecting the pill side with
    that ROI discards same-coloured material OUTSIDE the tray (e.g. a white
    counter behind white pills).  A final pass restores pills that are cropped
    by the frame edge (open notches the hole-fill cannot recover) when — and
    only when — the tray demonstrably wraps them, so the count is never
    inflated by off-tray background.
    """
    tray_core, roi = _largest_filled_regions(side_is_tray, 0.04, img_area)
    if cv2.countNonZero(roi) >= 0.12 * img_area:
        pill_mask = cv2.bitwise_and(side_is_pill, roi)
        border = _recover_border_pills(side_is_pill, roi, tray_core, img_area, ctx)
        if cv2.countNonZero(border):
            pill_mask = cv2.bitwise_or(pill_mask, border)
        return roi, pill_mask
    # Tray not found — treat the whole frame as ROI (colour/saturation only).
    full = np.full(side_is_tray.shape, 255, np.uint8)
    return full, side_is_pill


def _split_channel(chan: np.ndarray, tray_is_high: bool, img_area: float,
                   ctx: dict | None = None):
    """Otsu-split a single channel, then resolve tray ROI and pill mask."""
    chan = cv2.GaussianBlur(chan, (0, 0), 1.5)
    thr, _ = cv2.threshold(chan, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high = (chan > thr).astype(np.uint8) * 255
    low  = cv2.bitwise_not(high)
    if tray_is_high:
        return _tray_and_pills(high, low, img_area, ctx)
    return _tray_and_pills(low, high, img_area, ctx)


def _split_hue(hsv: np.ndarray, pill_hue: float, bg_hue: float, img_area: float,
               ctx: dict | None = None):
    """
    Separate two distinct saturated colours by nearest hue.

    Each pixel is assigned to whichever reference hue (pill or tray) it is closer
    to on the colour wheel.  Hue is invariant to brightness, so this captures
    shaded/glared parts of a coloured pill that a Hue-Saturation back-projection
    (which also keys on saturation) tends to drop.
    """
    h = hsv[:, :, 0].astype(np.float32)
    dp = np.abs(h - pill_hue); dp = np.minimum(dp, 180.0 - dp)
    db = np.abs(h - bg_hue);   db = np.minimum(db, 180.0 - db)
    pill_side = (dp < db).astype(np.uint8) * 255
    tray_side = (db <= dp).astype(np.uint8) * 255
    return _tray_and_pills(tray_side, pill_side, img_area, ctx)


def _split_value_flatfield(v: np.ndarray, pill_brighter: bool, img_area: float,
                           ctx: dict | None = None):
    """
    Separate pills from tray on the VALUE channel after flat-field correction.

    A plain global threshold on brightness fails when the tray has a lighting
    gradient or glare (the threshold ends up splitting bright-tray from
    dark-tray, not pill from tray).  Dividing by a heavily-blurred copy removes
    the smooth illumination AND smooth glare, leaving compact high-contrast
    objects (pills) standing out from a flat ~1.0 background — the robust way to
    find a white pill on a grey/steel tray.
    """
    h, w = v.shape
    vf = v.astype(np.float32)
    sigma = max(15.0, min(h, w) * 0.08)
    bg = cv2.GaussianBlur(vf, (0, 0), sigma)
    # Subtractive high-pass: remove the smooth illumination/glare, recenter at
    # 128.  Pills become compact deviations from a flat mid-grey background.
    corrected = np.clip(vf - bg + 128.0, 0, 255).astype(np.uint8)
    corrected = cv2.GaussianBlur(corrected, (0, 0), 1.5)
    thr, _ = cv2.threshold(corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if pill_brighter:
        pill_side = (corrected > max(thr, 132)).astype(np.uint8) * 255
    else:
        pill_side = (corrected < min(thr, 124)).astype(np.uint8) * 255
    # A dark embossed score line is below any pill/tray threshold and would
    # split a pill in two; a modest close bridges these thin lines without
    # fusing separate pills (it stays well under the inter-pill gap).
    ek = max(3, int(round(min(h, w) * 0.008))) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ek, ek))
    pill_side = cv2.morphologyEx(pill_side, cv2.MORPH_CLOSE, kernel, iterations=1)
    tray_side = cv2.bitwise_not(pill_side)
    return _tray_and_pills(tray_side, pill_side, img_area, ctx)


def _split_lab_distance(bgr: np.ndarray, hsv: np.ndarray, ref_shape: dict,
                        img_area: float, ctx: dict | None = None):
    """
    Separate pills from tray by LAB distance from the tray's reference colour.

    This is the only cue that sees a TWO-TONE pill as one object (both of its
    colours differ from the tray) and that combines hue, saturation and
    brightness into a single measure — the group-photo analogue of the
    corner-colour cue used on the reference photo.
    """
    px = np.uint8([[[round(ref_shape["bg_hue"]), round(ref_shape["bg_sat"]),
                     round(ref_shape["bg_val"])]]])
    tray_lab = cv2.cvtColor(cv2.cvtColor(px, cv2.COLOR_HSV2BGR),
                            cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0),
                       cv2.COLOR_BGR2LAB).astype(np.float32)
    dist = np.linalg.norm(lab - tray_lab, axis=2)
    dist_u = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, pill_side = cv2.threshold(dist_u, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tray_side = cv2.bitwise_not(pill_side)
    return _tray_and_pills(tray_side, pill_side, img_area, ctx)


def _dt_peak_stats(pill_mask: np.ndarray):
    """
    (n_consistent_peaks, half_width) of a mask's distance-transform peaks.

    Every pill contributes one interior peak whether or not it touches its
    neighbours, and the peak value is the pill's half-width.  Two rounds:
    estimate the typical half-width from the strongest peaks, then re-collect
    peaks non-max-suppressed at that scale.
    """
    dt = cv2.distanceTransform(pill_mask, cv2.DIST_L2, 5)
    if dt.max() < 3:
        return 0, 0.0
    r = float(np.percentile(dt[dt > 0], 99)) * 0.8
    vals = None
    for _ in range(2):
        size = max(9, int(round(1.6 * r)) | 1)
        mx = ndimage.maximum_filter(dt, size=size)
        py, px = np.nonzero((dt >= mx - 1e-3) & (dt >= 0.4 * r))
        if len(py) == 0:
            return 0, 0.0
        order = np.argsort(-dt[py, px])
        peaks = []
        for idx in order[:4000]:
            y, x = py[idx], px[idx]
            if all((y - oy) ** 2 + (x - ox) ** 2 > (1.5 * r) ** 2
                   for oy, ox in peaks):
                peaks.append((y, x))
            if len(peaks) >= 800:
                break
        vals = np.array([dt[y, x] for (y, x) in peaks], np.float32)
        r = float(np.median(vals))
    n = int(np.sum((vals >= 0.72 * r) & (vals <= 1.38 * r)))
    return n, r


def _mask_quality(pill_mask: np.ndarray, img_area: float):
    """
    Measure how much a candidate pill mask looks like "a repeated population
    of similar-size objects".

    Isolated-component counting fails exactly when it matters (touching or
    blurred pills leave no clean singletons), so pills are counted as PEAKS of
    the distance transform: every pill contributes one interior peak whether
    or not it touches its neighbours, and the peak value is the pill's
    half-width.  A complete mask yields many peaks of consistent half-width; a
    fragmentary mask yields few; an inverted/degenerate mask yields wildly
    inconsistent ones.

    Returns (n_consistent_peaks, mode_area); (0, 0) for degenerate masks.
    """
    if pill_mask is None:
        return (0, 0.0)
    fill = cv2.countNonZero(pill_mask) / img_area
    # Even a tray packed edge-to-edge with pills stays under ~60% of the
    # frame; more foreground than that means the polarity is inverted.
    if fill < 0.0015 or fill > 0.60:
        return (0, 0.0)
    n, r = _dt_peak_stats(pill_mask)
    if n == 0:
        return (0, 0.0)
    mode_area = float(np.pi * r * r)
    # Consistent SPECKLE (embossing fragments, glare dots) also repeats, but
    # real pills in a group photo are never this small a fraction of the frame.
    if mode_area < 0.0008 * img_area:
        return (0, 0.0)
    # The consistent objects must account for a reasonable share of the mask:
    # a mask that covers half the frame but contains only six pill-sized peaks
    # is background noise that happens to include the pills, not a pill mask.
    if n * mode_area < 0.12 * fill * img_area:
        return (0, 0.0)
    # Rank by explained pill MASS, not peak count: a fragment mask shatters
    # each pill into several smaller consistent shards and would win a pure
    # peak-count comparison, while explaining far fewer pill pixels.  Weight
    # by precision so that of two masks explaining the same pills, the one
    # without extra junk (shadows, background bleed) wins.
    mass = n * mode_area
    precision = min(1.0, mass / (fill * img_area))
    return (mass * precision, mode_area)


def _make_ctx(group_bgr: np.ndarray, rs: dict) -> dict | None:
    """LAB image + learned pill/tray colour models, shared by the appearance
    checks (border-pill recovery, thick-cluster validation)."""
    keys = ("bg_hue", "bg_sat", "bg_val", "pill_hue", "pill_sat", "pill_val")
    if not all(k in rs for k in keys):
        return None

    def _hsv_to_lab(h_, s_, v_):
        px = np.uint8([[[int(round(h_)) % 180, int(np.clip(s_, 0, 255)),
                         int(np.clip(v_, 0, 255))]]])
        return cv2.cvtColor(cv2.cvtColor(px, cv2.COLOR_HSV2BGR),
                            cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]

    return {
        "labf": cv2.cvtColor(group_bgr, cv2.COLOR_BGR2LAB).astype(np.float32),
        "tray_lab": _hsv_to_lab(rs["bg_hue"], rs["bg_sat"], rs["bg_val"]),
        "pill_lab": _hsv_to_lab(rs["pill_hue"], rs["pill_sat"], rs["pill_val"]),
    }


def _build_pill_mask(group_bgr: np.ndarray,
                     pill_hist: np.ndarray,
                     bg_hist: np.ndarray,
                     is_achromatic: bool,
                     ref_shape: dict | None = None):
    """
    Return (pill_mask, roi).

    The discriminator that best separates the pill from its tray is chosen from
    the reference statistics, then ALL paths share the same robust logic: the
    tray is the dominant region of the "tray side", the ROI is that region with
    its holes filled (recovering the pills resting on it), and the pills are the
    "pill side" intersected with the ROI (so same-coloured material OUTSIDE the
    tray, e.g. a white counter, is discarded).

    * SATURATION — pale pill on a coloured tray, or coloured pill on a grey
      tray.  Ignores hue, so it survives a glossy tray's glare gradient.
    * HUE — two saturated colours (e.g. orange on blue), via a per-pixel
      nearest-hue classifier (invariant to shading).
    * VALUE — achromatic pill on achromatic tray (white on grey, orange on dark
      wood), via flat-field correction that removes illumination and glare.

    Hue-Saturation back-projection is the final fallback when none of the three
    cleanly separates.
    """
    h, w = group_bgr.shape[:2]
    img_area = float(h * w)
    hsv = cv2.cvtColor(group_bgr, cv2.COLOR_BGR2HSV)

    rs = ref_shape or {}
    pill_sat = rs.get("pill_sat", _hist_mean_sat(pill_hist))
    bg_sat   = rs.get("bg_sat",   _hist_mean_sat(bg_hist))
    pill_val = rs.get("pill_val", 128.0)
    bg_val   = rs.get("bg_val",   128.0)
    pill_hue = rs.get("pill_hue")
    bg_hue   = rs.get("bg_hue")
    sat_sep  = abs(pill_sat - bg_sat)
    val_sep  = abs(pill_val - bg_val)
    # Hue is meaningful whenever both sides carry SOME colour; the mask-quality
    # comparison below discards a hue split that turns out to be noise, so the
    # gate can be permissive (a strict gate here cost real separations, e.g.
    # a slightly desaturated blue pill on a beige tray).
    both_saturated = pill_sat >= 40 and bg_sat >= 40
    hue_sep = (_hue_dist(pill_hue, bg_hue)
               if (pill_hue is not None and bg_hue is not None and both_saturated) else 0.0)

    roi = np.full((h, w), 255, np.uint8)
    pill_mask = None

    # Cue preference by normalised separation of the reference statistics.
    # Saturation gets a bonus because it is invariant to a glossy tray's
    # glare/brightness gradient; it is trustworthy only when one side is
    # genuinely saturated.
    sat_reliable = sat_sep >= 30 and max(pill_sat, bg_sat) >= 70
    sat_score = (sat_sep / 255.0) * (1.6 if sat_reliable else 0.6)
    val_score = (val_sep / 255.0)
    hue_score = (hue_sep / 90.0) if hue_sep > 0 else 0.0

    # Build ALL candidate masks and choose by MEASURED quality (how much each
    # mask looks like a repeated population of pill-shaped objects) rather
    # than committing to a single cue.  Reference statistics can be subtly
    # wrong (contaminated background, unusual lighting) and a mis-chosen cue
    # produces a degenerate mask — invisible unless masks are compared.
    # Colour models for border-pill recovery: a component at the frame edge is
    # only admitted when the surface wrapping it looks like the LEARNED tray.
    ctx = _make_ctx(group_bgr, rs)

    candidates = []          # (pref, roi, mask)
    if sat_score >= 0.05:
        r_, m_ = _split_channel(hsv[:, :, 1], bg_sat > pill_sat, img_area, ctx)
        candidates.append((sat_score + 0.02, r_, m_))
    if hue_score > 0:
        r_, m_ = _split_hue(hsv, pill_hue, bg_hue, img_area, ctx)
        candidates.append((hue_score + 0.01, r_, m_))
    if val_sep >= 8:
        r_, m_ = _split_value_flatfield(hsv[:, :, 2], pill_val > bg_val,
                                        img_area, ctx)
        candidates.append((val_score, r_, m_))
    if all(k in rs for k in ("bg_hue", "bg_sat", "bg_val")):
        r_, m_ = _split_lab_distance(group_bgr, hsv, rs, img_area, ctx)
        candidates.append((0.05, r_, m_))
    # Histogram back-projection: the last-resort discriminator.
    p_pill = _backproject(hsv, pill_hist).astype(np.int16)
    p_bg   = _backproject(hsv, bg_hist).astype(np.int16)
    blur = lambda m: cv2.GaussianBlur(m.astype(np.float32), (0, 0), 2.0)
    fp, fb = blur(p_pill), blur(p_bg)
    tray_side = ((fb > fp) & (fb > 15)).astype(np.uint8) * 255
    pill_side = (fp >= fb).astype(np.uint8) * 255
    r_, m_ = _tray_and_pills(tray_side, pill_side, img_area, ctx)
    candidates.append((0.0, r_, m_))

    best_key = None
    for pref, cand_roi, cand_mask in candidates:
        n, mode_area = _mask_quality(cand_mask, img_area)
        key = (n, mode_area, pref)
        if best_key is None or key > best_key:
            best_key = key
            roi, pill_mask = cand_roi, cand_mask

    if pill_mask is None or cv2.countNonZero(pill_mask) < 0.0015 * img_area:
        pill_mask = (fp > fb).astype(np.uint8) * 255

    # Degenerate guard: nearly-empty or nearly-full mask -> last-resort.
    fill = cv2.countNonZero(pill_mask) / img_area
    if fill < 0.0008 or fill > 0.92:
        pill_mask = _achromatic_mask(group_bgr)

    return pill_mask, roi


def _achromatic_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Last-resort mask for white/grey pills: bright, low-saturation pixels.
    Retained as a safety net for the degenerate case where colour
    back-projection cannot separate the pill from its surroundings.
    """
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    _, v_hi = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sat_gate = np.uint8(s < 60) * 255
    mask = cv2.bitwise_and(v_hi, sat_gate)
    if cv2.countNonZero(mask) / mask.size > 0.5:
        mask = cv2.bitwise_not(mask)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Scale estimation and cleanup
# ─────────────────────────────────────────────────────────────────────────────

def _component_metrics(mask: np.ndarray):
    """Yield (comp_mask, pixel_area, contour, shape_metrics) per component."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        comp = np.uint8(labels == i) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        out.append((comp, area, c, _shape_metrics(c)))
    return out


def _estimate_pill_area(mask: np.ndarray, ref_area: float,
                        ref_shape: dict | None) -> float:
    """
    Estimate the area of ONE pill from isolated, solid, pill-shaped components
    ("singletons") in the group mask.  This auto-corrects an unreliable
    reference area and adapts to the group photo's camera distance.
    """
    comps = _component_metrics(mask)
    if not comps:
        return ref_area

    # A trustworthy single pill is solid AND genuinely pill-SHAPED (round/oval).
    # Requiring circularity is what separates real pills from glare streaks,
    # speckle clusters and merged blobs — all of which are irregular.
    if ref_shape is not None:
        circ_min = max(0.42, ref_shape["circularity"] - 0.22)
        # Generous: the reference aspect ratio is measured from one contour
        # and can be off by 0.3 for oblongs; solidity + circularity already
        # reject glare streaks, so this gate only needs to catch extremes.
        ar_min   = max(0.15, ref_shape["aspect_ratio"] - 0.40)
    else:
        circ_min, ar_min = 0.55, 0.30

    # The speck floor must NOT scale with the reference: a close-up reference
    # pill can be 25x the group pills' size, and a ref-relative floor would
    # then exclude every real pill (leaving no singletons and a catastrophic
    # eff_area = ref_area -> count 0).  Speck size is a property of the image.
    singletons = sorted(
        area for (_, area, _, sm) in comps
        if area >= max(200, 0.0003 * mask.size)
        and sm["solidity"] >= 0.90
        and sm["circularity"] >= circ_min
        and sm["aspect_ratio"] >= ar_min
    )

    if singletons:
        # Real pills of one type REPEAT at the same size; sporadic contaminants
        # (embossing, glints, half-pill fragments) do not.  So group the areas
        # into size clusters (consecutive ratio <= 1.5) and trust the most
        # populous cluster — on a tie prefer the larger pills.  This is far more
        # robust than a percentile, which a single small contaminant skews.
        clusters = [[singletons[0]]]
        for a in singletons[1:]:
            if a <= clusters[-1][-1] * 1.5:
                clusters[-1].append(a)
            else:
                clusters.append([a])
        best = max(clusters, key=lambda c: (len(c), sum(c)))
        est = float(np.median(best))
        # Noise guard: an estimate far below the reference is either a real
        # camera-distance change (reference shot close up) or speckle noise.
        # Real pills leave MANY consistent singletons and clusters of sane
        # size; speckle leaves a few tiny blobs next to a giant merged blob
        # (dividing by which would multiply the count catastrophically).  So a
        # small estimate is trusted only with strong, consistent evidence.
        if est < ref_area * 0.12:
            largest = max(area for (_, area, _, _) in comps)
            if len(best) < 4 or largest > est * 80:
                return ref_area
        return float(np.clip(est, ref_area / 25.0, ref_area * 15.0))

    # No clean singleton: trust the reference.  Deriving a scale from whatever
    # solid blob happens to be smallest lets a single glint/speck set the pill
    # size and multiply every cluster's count by 10-15x (a catastrophic
    # overcount).  Without singleton evidence, the reference area — measured
    # from a real pill — is the only trustworthy scale.
    return ref_area


def _clean_mask(mask: np.ndarray, eff_radius: float) -> np.ndarray:
    """
    Open to drop specks, close to fill embossing/glare holes — kernels scaled to
    the pill radius.  The close kernel is kept small so touching pills are not
    fused beyond what the watershed can separate.
    """
    r_open  = max(1, int(round(eff_radius * 0.08)))
    r_close = max(1, int(round(eff_radius * 0.06)))   # tiny: fill embossing only
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_open * 2 + 1,) * 2)
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_close * 2 + 1,) * 2)
    # Close first (fill engraving holes inside a pill), then open (drop specks
    # and sever thin bridges between near pills).  Both kernels are kept well
    # below the inter-pill gap so separate pills are never fused — touching
    # pills are split later by the distance-transform watershed.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  ko, iterations=1)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Counting (connected components + area ratio)
# ─────────────────────────────────────────────────────────────────────────────

def _count_pills_in_mask(mask: np.ndarray, annotated: np.ndarray,
                         eff_area: float, eff_radius: float,
                         ref_shape: dict | None, ctx: dict | None = None):
    """
    Count pills in a cleaned binary mask.

    Each connected component is either one pill (single-pill-sized) or a cluster
    of touching pills.  For clusters the AREA ratio is the estimate — for flat,
    non-overlapping pills total area = N x pill area.  Single-sized blobs are
    shape-validated to reject stray noise.

    Returns (total_count, annotated, pill_regions) with pill_regions a list of
    (contour, pixel_area, count) for annotation.
    """
    if cv2.countNonZero(mask) == 0:
        return 0, annotated, []

    h, w = mask.shape
    num, comp_labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_area = max(60, eff_area * 0.30)

    total = 0
    regions = []
    for ci in range(1, num):
        area = int(stats[ci, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x  = stats[ci, cv2.CC_STAT_LEFT]
        y  = stats[ci, cv2.CC_STAT_TOP]
        bw = stats[ci, cv2.CC_STAT_WIDTH]
        bh = stats[ci, cv2.CC_STAT_HEIGHT]
        touches_border = (x == 0 or y == 0 or x + bw >= w or y + bh >= h)
        comp = np.uint8(comp_labels == ci) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        contour = max(cnts, key=cv2.contourArea)
        metrics = _shape_metrics(contour)
        area_n = max(1, int(round(area / eff_area)))

        if area <= eff_area * 1.55:
            # Single pill — validate shape to reject stray noise blobs.  Accept
            # either a close match to the reference shape OR a solid, convex,
            # full-pill-sized blob (segmentation can make a real pill slightly
            # non-circular).  The size gate (>= 0.6x pill area) is what stops
            # half-pill fragments from being double-counted.
            convex_full = (metrics["solidity"] >= 0.85 and
                           metrics["aspect_ratio"] >= 0.20 and
                           area >= 0.6 * eff_area)
            # A pill cropped by the frame edge is TRUNCATED: it legitimately has
            # less area and a cut-off (lower-circularity) outline, so the full-
            # pill gates above wrongly drop it.  Any border-touching component
            # that reaches this stage already survived the strict tray-surround
            # and size test in _recover_border_pills, so accepting a solid,
            # convex cropped blob as one pill cannot admit glare/background —
            # it only restores genuinely cropped pills (never an overcount).
            cropped_pill = (touches_border and
                            metrics["solidity"] >= 0.80 and
                            metrics["aspect_ratio"] >= 0.20)
            if not (convex_full or cropped_pill or
                    _shape_is_pill_like(metrics, ref_shape, strict=True)):
                continue
            count = 1
        else:
            # Cluster of touching pills.  Reject only thin/ragged non-pill
            # streaks (true pill clusters stay reasonably solid); the glare /
            # background is already excluded by the saturation/tray mask.
            if metrics["solidity"] < 0.40:
                continue
            # A cluster of single-layer pills is roughly one pill THICK: its
            # distance-transform peak sits near the pill radius.  A ragged
            # background network has no pill-thick interior (peak far below)
            # and is never a pill cluster.  A region much THICKER than a pill
            # is either a background sheet or a dense gap-free pile — geometry
            # cannot tell them apart, so appearance decides: keep it only if
            # its interior actually looks like the pill rather than the tray.
            comp_dt = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
            peak = float(comp_dt.max())
            if peak < 0.45 * eff_radius:
                continue
            if peak > 2.4 * eff_radius:
                # Off-tray background always reaches the frame border (it
                # surrounds the tray); a dense pile of pills sits interior.
                border_px = (int(np.count_nonzero(comp[0, :])) +
                             int(np.count_nonzero(comp[-1, :])) +
                             int(np.count_nonzero(comp[:, 0])) +
                             int(np.count_nonzero(comp[:, -1])))
                if border_px > 10 or ctx is None:
                    continue
                comp_med = np.median(ctx["labf"][comp > 0], axis=0)
                d_pill = float(np.linalg.norm(comp_med - ctx["pill_lab"]))
                d_tray = float(np.linalg.norm(comp_med - ctx["tray_lab"]))
                if d_pill >= d_tray:
                    continue
            # Area ratio is the reliable estimate for flat, non-overlapping pills
            # (total area = N x pill area).  Distance-transform peaks over-segment
            # capsules (a long ridge yields several maxima), so peaks refine the
            # count only for ROUND pills, where one pill = one peak and the
            # estimate is immune to interstitial-gap inflation and to an
            # imprecise reference size.  Wildly inconsistent peak counts mean
            # the cluster is not clean circles after all — keep the area ratio.
            count = area_n

        total += count
        regions.append((contour, area, count))

    return total, annotated, regions


# ─────────────────────────────────────────────────────────────────────────────
# Annotation
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE = [
    (34, 197, 94), (59, 130, 246), (249, 115, 22), (168, 85, 247),
    (20, 184, 166), (234, 179, 8), (239, 68, 68),
]


def _annotate(annotated: np.ndarray, pill_regions: list, total_count: int) -> np.ndarray:
    h, w = annotated.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, (contour, area, pill_count) in enumerate(pill_regions):
        bgr   = _PALETTE[i % len(_PALETTE)]
        color = (int(bgr[2]), int(bgr[1]), int(bgr[0]))

        overlay = annotated.copy()
        cv2.drawContours(overlay, [contour], -1, color, -1)
        cv2.addWeighted(overlay, 0.18, annotated, 0.82, 0, annotated)
        cv2.drawContours(annotated, [contour], -1, color, 2)

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        else:
            rx, ry, rw, rh = cv2.boundingRect(contour)
            cx, cy = rx + rw // 2, ry + rh // 2

        label = str(pill_count)
        fs = min(1.0, max(0.4, np.sqrt(area / (h * w)) * 8))
        thickness = max(1, int(fs * 2))
        (tw, th), _ = cv2.getTextSize(label, font, fs, thickness)
        pad = 4
        cv2.rectangle(annotated,
                      (max(0, cx - tw // 2 - pad), max(0, cy - th // 2 - pad)),
                      (min(w - 1, cx + tw // 2 + pad), min(h - 1, cy + th // 2 + pad)),
                      color, -1)
        cv2.putText(annotated, label, (cx - tw // 2, cy + th // 2),
                    font, fs, (0, 0, 0), thickness, cv2.LINE_AA)

    banner = f"Total: {total_count}"
    fs_b = 1.2
    (tw, th), _ = cv2.getTextSize(banner, font, fs_b, 2)
    cv2.rectangle(annotated, (8, 8), (tw + 22, th + 22), (20, 20, 20), -1)
    cv2.putText(annotated, banner, (14, th + 14), font, fs_b, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated


# ─────────────────────────────────────────────────────────────────────────────
# Public API: counting
# ─────────────────────────────────────────────────────────────────────────────

def count_pills(group_image_np, ref_area, pill_hist, bg_hist,
                is_achromatic=False, ref_shape=None):
    """
    Count pills in the group image.

    Returns (count, annotated_bgr_image).
    """
    if group_image_np is None or group_image_np.size == 0:
        raise ValueError("Group image is empty or invalid.")

    group_image_np = _normalise_channels(group_image_np)
    annotated = group_image_np.copy()

    pill_mask, _roi = _build_pill_mask(
        group_image_np, pill_hist, bg_hist, is_achromatic, ref_shape
    )

    # Light, scale-independent despeckle ONLY — it must not close gaps between
    # pills, or isolated pills would merge and the scale estimate (which relies
    # on isolated singletons) would collapse.  Then estimate the true single-pill
    # size from the group photo and do the full, scale-aware cleanup.
    h, w = pill_mask.shape
    kp = max(2, int(round(min(h, w) * 0.0025))) | 1
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kp, kp))
    pre = cv2.morphologyEx(pill_mask, cv2.MORPH_OPEN, small, iterations=1)

    eff_area   = _estimate_pill_area(pre, ref_area, ref_shape)
    eff_radius = float(np.sqrt(max(eff_area, 1.0) / np.pi))

    mask = _clean_mask(pre, eff_radius)

    total, annotated, regions = _count_pills_in_mask(
        mask, annotated, eff_area, eff_radius, ref_shape,
        ctx=_make_ctx(group_image_np, ref_shape or {})
    )

    annotated = _annotate(annotated, regions, total)
    return total, annotated
