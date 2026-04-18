# Pill Counter Redesign — Histogram Backprojection Approach

**Date:** 2026-04-18  
**Status:** Approved  
**Scope:** Rework `pill_counter.py` to replace multi-mask competition with histogram backprojection for more reliable pill detection across all shapes, colors, and backgrounds.

---

## Problem Statement

The current implementation fails in two primary ways:
- **Wildly wrong counts in both directions** — over-counting (background texture detected as pills) and under-counting (pills merging or not detected)
- **Colored pill failures** — color mask and Otsu strategies compete; the heuristic mask scorer picks the wrong winner

Root cause: the multi-mask competition strategy (`_mask_score`) is fragile. It picks among ~5 candidate masks using a heuristic that can fail on unusual pill/background combinations.

---

## Constraints

- Two-image input is preserved: one reference pill image + one group photo
- Pure OpenCV, no ML dependencies added
- Must handle any pill color, shape, size, and any background
- Accuracy is the primary goal; performance is secondary

---

## Architecture

The redesigned pipeline has three stages:

```
Reference Image
    └─► GrabCut Extraction ──► Pill HS Histogram + Background HS Histogram
                             └► Shape Metrics (circularity, aspect, solidity, area)

Group Image
    └─► Backprojection (using ratio histogram) ──► Probability Map
                                               └─► Otsu Threshold ──► Binary Mask
                                                                   └─► Morphological Cleanup
                                                                       └─► Scale Estimation
                                                                           └─► Watershed Count
                                                                               └─► Final Count + Annotation
```

---

## Section 1: Reference Pill Extraction (`analyze_reference`)

**Goal:** Reliably isolate the pill from the reference image and extract its color distribution.

**Steps:**

1. **GrabCut initialization** — define a rectangle covering the central 60% of the image as probable foreground. Run GrabCut for 5 iterations. This uses a GMM internally and is robust to any pill color vs. any background.

2. **Contour selection** — from the GrabCut mask, extract contours. Select the contour whose centroid is closest to the image center (not the largest contour). Apply sanity filters:
   - Area must be 2%–85% of total image area
   - Solidity must be > 0.3

3. **Fallback** — if GrabCut produces no valid contour near center, fall back to the existing contour-selection logic (max solidity among size-filtered contours).

4. **Color histogram extraction** — compute a 2D Hue-Saturation histogram (32×32 bins) from HSV pixels inside the GrabCut mask (pill pixels). Also compute the same histogram for pixels outside the mask (background pixels).

5. **Shape metrics** — compute circularity, aspect ratio (minor/major from fitted ellipse), and solidity from the selected contour. These are used downstream for shape validation.

6. **Achromatic detection** — if >80% of pill pixels have saturation < 30, set `is_achromatic = True`. This triggers a different pipeline path for white/grey pills.

**Outputs:**
- `pill_hist`: 32×32 HS histogram (pill pixels)
- `bg_hist`: 32×32 HS histogram (background pixels)
- `ref_area`: pixel area of pill contour
- `ref_shape`: dict of circularity, aspect_ratio, solidity
- `is_achromatic`: bool

---

## Section 2: Probability Map Generation (`_build_probability_mask`)

**Goal:** Produce a reliable binary mask of pill pixels in the group image.

### Chromatic Path (is_achromatic = False)

1. **Ratio histogram** — divide `pill_hist` by (`bg_hist` + epsilon) bin-wise. Normalize so max value = 255. This gives high values to hue-saturation combinations that appear in pills but not in the background.

2. **Backprojection** — for every pixel in the group image, convert to HSV, look up its (H, S) bin in the ratio histogram. Assign that value as the pixel's probability. Result: grayscale probability map.

3. **Smooth** — apply a disc-shaped mean filter with radius = 10% of reference pill radius. Connects nearby pill pixels, suppresses isolated noise.

4. **Threshold** — apply Otsu's method to the probability map to find the natural split between pill-probability and background-probability pixels. Result: binary mask.

5. **Degenerate check** — if the mask covers <1% or >90% of the image area, fall back to LAB Otsu on the group image (best single strategy from old code).

### Achromatic Path (is_achromatic = True)

1. Convert group image to LAB
2. Apply CLAHE to the L channel (clipLimit=2.5, tileGridSize=8×8)
3. Apply Otsu's threshold to the enhanced L channel
4. Try both the mask and its inverse; pick the one where blob sizes better match reference area
5. This is essentially the current white-pill path, isolated and kept because it works well

---

## Section 3: Morphological Cleanup

Applied to the binary mask from Section 2, identical for both paths:

1. **Open** with elliptical kernel of radius = 12% of reference pill radius — removes noise smaller than a pill
2. **Close** with elliptical kernel of radius = 8% of reference pill radius — fills holes inside pills without merging adjacent ones

Kernel sizes are clamped to minimum radius of 2px to avoid degenerate kernels on very small reference pills.

---

## Section 4: Scale Estimation (`_estimate_scale`)

1. Find all contours in the cleaned binary mask
2. Collect blobs with area between 20% and 300% of `ref_area` (single-pill candidates)
3. Use the **median** blob area as the scale estimate (`scaled_ref_area`)
4. Fallback: if no blobs in range, use the smallest blob that is at least 15% of `ref_area`

---

## Section 5: Watershed Counting (`_watershed_count`)

1. **Distance transform** on binary mask
2. For each connected component separately, find local maximum of distance transform, set foreground threshold at **0.35 × local_max** (per-component, not global)
3. Build watershed markers from sure-foreground connected components
4. Run watershed
5. For each labeled region: `pill_count = round(region_area / scaled_ref_area)`

**Sanity check via local maxima:**
- Find local maxima of the distance transform (window = 40% of reference pill radius, min height = 28% of peak)
- If `local_maxima_count >= 2 × area_count`, increment `area_count` by 1 (catches severe over-merging)

**Shape validation (single-pill regions only):**
- Circularity must be within ±0.40 of reference (relaxed from ±0.30)
- Aspect ratio must be within ±0.40 of reference
- Solidity must be within ±0.45 of reference
- Regions failing all three shape checks are discarded

**Fallback:**
- If watershed finds zero pills, run plain contour analysis with same area-ratio counting

---

## Section 6: Error Handling Summary

| Failure Mode | Detection | Response |
|---|---|---|
| GrabCut finds no pill near center | No valid contour within 30% of image diagonal from center | Fall back to max-solidity contour selection |
| Backprojection mask degenerate (<1% or >90%) | Area check after threshold | Fall back to LAB Otsu |
| Achromatic pill | >80% of pill pixels with saturation <30 | Use achromatic path (CLAHE + LAB Otsu) |
| No pills found after watershed | Count = 0 | Run contour fallback, return that count |

---

## What Is Removed

- Multi-mask competition (`_mask_score`, 5 candidate mask strategies)
- Reference-calibrated LAB projection mask
- K-means color clustering (replaced by HS histogram)
- Adaptive global watershed foreground threshold (replaced by per-component threshold)
- Surface ROI detection (added complexity without consistent benefit)

---

## Files Changed

- `pill_counter.py` — full rewrite of `analyze_reference` and `count_pills` internals; `app.py` interface unchanged
- `test_pill.py` — add tests for colored pills on varied backgrounds

---

## Success Criteria

- Correct count (within ±1) on white oval pills on blue tray
- Correct count (within ±1) on orange round pills on gray tray
- Correct count on at least one additional pill/background combination not in current test suite
- No regression on existing test images
