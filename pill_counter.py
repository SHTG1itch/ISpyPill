import cv2
import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_reference(image_np):
    """
    Analyze a single reference pill image to extract:
      - ref_area      : float  — pixel area of one pill
      - color_profiles: list   — one dict per dominant color cluster
      - is_white      : bool   — True if pill is fully achromatic (white/gray)
      - ref_shape     : dict   — circularity, aspect_ratio, solidity

    Each color_profile dict has keys:
        hsv_lower, hsv_upper  (np.float32 arrays, shape (3,))
        is_achromatic         (bool)
        wraps_hue             (bool)
        weight                (float, fraction of pill pixels in this cluster)
        mean_value            (float, V-channel mean for brightness fallback)
    """
    if image_np is None or image_np.size == 0:
        raise ValueError("Reference image is empty or invalid.")

    # Normalise to 3-channel BGR
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    elif image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2BGR)

    h, w = image_np.shape[:2]

    # ------------------------------------------------------------------
    # Step 1: Isolate pill using Otsu thresholding
    # ------------------------------------------------------------------
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_mask_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)

    # ------------------------------------------------------------------
    # Step 2: Find the reference pill contour
    #
    # Strategy: collect candidates from both Otsu variants, keep only those
    # with area in [1%, 85%] of the image (excludes background and tiny noise),
    # then pick the most convex one (pills are convex; background frames are not).
    # ------------------------------------------------------------------
    min_area_frac = 0.01
    max_area_frac = 0.85
    img_area = h * w
    all_candidates = []  # list of (contour, solidity)

    for raw_mask in [otsu_mask, otsu_mask_inv]:
        cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  kernel, iterations=1)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area_frac = cv2.contourArea(c) / img_area
            if min_area_frac <= area_frac <= max_area_frac:
                hull_area = cv2.contourArea(cv2.convexHull(c)) + 1e-6
                solidity = cv2.contourArea(c) / hull_area
                all_candidates.append((c, solidity))

    if not all_candidates:
        # Fallback: adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2
        )
        adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area_frac = cv2.contourArea(c) / img_area
            if min_area_frac <= area_frac <= max_area_frac:
                hull_area = cv2.contourArea(cv2.convexHull(c)) + 1e-6
                solidity = cv2.contourArea(c) / hull_area
                all_candidates.append((c, solidity))

    if not all_candidates:
        raise ValueError(
            "Could not isolate the reference pill. "
            "Please use a photo with the pill clearly visible against a contrasting background."
        )

    # Most convex candidate = pill (pills are compact/convex; background frames are not)
    ref_contour = max(all_candidates, key=lambda x: x[1])[0]
    ref_area = cv2.contourArea(ref_contour)

    if ref_area < 100:
        raise ValueError("Reference pill appears too small in the image. Please use a closer photo.")

    # ------------------------------------------------------------------
    # Step 3: Shape features
    # ------------------------------------------------------------------
    perimeter = cv2.arcLength(ref_contour, True)
    circularity = (4 * np.pi * ref_area) / (perimeter ** 2 + 1e-6)

    if len(ref_contour) >= 5:
        ellipse = cv2.fitEllipse(ref_contour)
        ax1, ax2 = ellipse[1]
        aspect_ratio = min(ax1, ax2) / (max(ax1, ax2) + 1e-6)
    else:
        rx, ry, rw, rh = cv2.boundingRect(ref_contour)
        aspect_ratio = min(rw, rh) / (max(rw, rh) + 1e-6)

    hull = cv2.convexHull(ref_contour)
    hull_area = cv2.contourArea(hull)
    solidity = ref_area / (hull_area + 1e-6)

    ref_shape = {
        "circularity": float(circularity),
        "aspect_ratio": float(aspect_ratio),
        "solidity": float(solidity),
    }

    # ------------------------------------------------------------------
    # Step 4: Extract HSV pixels inside pill region
    # ------------------------------------------------------------------
    pill_region_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(pill_region_mask, [ref_contour], -1, 255, -1)

    hsv_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
    pill_hsv_pixels = hsv_image[pill_region_mask == 255].astype(np.float32)  # (N, 3)

    if len(pill_hsv_pixels) == 0:
        raise ValueError("Could not extract color from reference pill region.")

    # ------------------------------------------------------------------
    # Step 5: K-Means color clustering
    # ------------------------------------------------------------------
    mean_saturation = float(np.mean(pill_hsv_pixels[:, 1]))
    is_fully_achromatic = mean_saturation < 40

    if is_fully_achromatic or len(pill_hsv_pixels) < 20:
        # Single cluster for white/gray pills
        k_best = 1
        best_labels = np.zeros(len(pill_hsv_pixels), dtype=np.int32)
        best_centers = np.mean(pill_hsv_pixels, axis=0, keepdims=True)
    else:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        comp2, labels2, centers2 = cv2.kmeans(
            pill_hsv_pixels, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )
        comp3, labels3, centers3 = cv2.kmeans(
            pill_hsv_pixels, 3, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )
        # Use K=3 only if it reduces variance by >= 40%
        if comp2 > 0 and (comp3 / comp2) < 0.6:
            k_best = 3
            best_labels = labels3.ravel()
            best_centers = centers3
        else:
            k_best = 2
            best_labels = labels2.ravel()
            best_centers = centers2

    # ------------------------------------------------------------------
    # Step 6: Build one color profile per cluster
    # ------------------------------------------------------------------
    color_profiles = []
    total_pixels = len(pill_hsv_pixels)

    for k in range(k_best):
        mask_k = best_labels == k
        cluster_pixels = pill_hsv_pixels[mask_k]
        if len(cluster_pixels) == 0:
            continue

        weight = len(cluster_pixels) / total_pixels
        # Skip clusters that represent < 5% of the pill (noise)
        if weight < 0.05 and k_best > 1:
            continue

        center = best_centers[k] if k_best > 1 else best_centers[0]
        std_hsv = np.std(cluster_pixels, axis=0)

        h_tol = max(15.0, 2.0 * std_hsv[0])
        s_tol = max(50.0, 2.0 * std_hsv[1])
        v_tol = max(50.0, 2.0 * std_hsv[2])

        hsv_lower = np.array([
            max(0.0,   center[0] - h_tol),
            max(0.0,   center[1] - s_tol),
            max(0.0,   center[2] - v_tol),
        ], dtype=np.float32)

        hsv_upper = np.array([
            min(179.0, center[0] + h_tol),
            min(255.0, center[1] + s_tol),
            min(255.0, center[2] + v_tol),
        ], dtype=np.float32)

        is_achromatic = bool(center[1] < 40)
        wraps_hue = bool(center[0] < 10 or center[0] > 165)

        color_profiles.append({
            "hsv_lower":    hsv_lower,
            "hsv_upper":    hsv_upper,
            "is_achromatic": is_achromatic,
            "wraps_hue":    wraps_hue,
            "weight":       float(weight),
            "mean_value":   float(center[2]),
        })

    if not color_profiles:
        # Degenerate fallback — treat entire pill as single achromatic cluster
        mean_v = float(np.mean(pill_hsv_pixels[:, 2]))
        color_profiles = [{
            "hsv_lower":    np.array([0, 0, max(0.0, mean_v - 60)], dtype=np.float32),
            "hsv_upper":    np.array([179, 255, min(255.0, mean_v + 60)], dtype=np.float32),
            "is_achromatic": True,
            "wraps_hue":    False,
            "weight":       1.0,
            "mean_value":   mean_v,
        }]

    is_white = all(p["is_achromatic"] for p in color_profiles)

    return float(ref_area), color_profiles, is_white, ref_shape


def count_pills(group_image_np, ref_area, color_profiles, is_white=False, ref_shape=None):
    """
    Count pills in a group image using reference characteristics.

    Returns:
        count     (int)        : total number of pills detected
        annotated (np.ndarray) : BGR image with pills highlighted
    """
    if group_image_np is None or group_image_np.size == 0:
        raise ValueError("Group image is empty or invalid.")

    if len(group_image_np.shape) == 2:
        group_image_np = cv2.cvtColor(group_image_np, cv2.COLOR_GRAY2BGR)
    elif group_image_np.shape[2] == 4:
        group_image_np = cv2.cvtColor(group_image_np, cv2.COLOR_BGRA2BGR)

    h, w = group_image_np.shape[:2]
    annotated = group_image_np.copy()

    # Derive mean_ref_value for brightness fallback
    mean_ref_value = max(p["mean_value"] for p in color_profiles)

    blurred = cv2.GaussianBlur(group_image_np, (5, 5), 0)

    # ------------------------------------------------------------------
    # Build pill mask
    # ------------------------------------------------------------------
    if is_white:
        mask = _white_pill_mask(blurred, mean_ref_value)
    else:
        mask = _color_pill_mask(blurred, color_profiles)

    # Quality guard
    mask_fill_ratio = np.sum(mask > 0) / (h * w)

    if mask_fill_ratio < 0.01:
        # Color mask too narrow — fall back to brightness
        mask = _white_pill_mask(blurred, mean_ref_value)
        mask_fill_ratio = np.sum(mask > 0) / (h * w)

    if mask_fill_ratio > 0.90:
        # Mask covers almost everything — probably inverted
        mask = cv2.bitwise_not(mask)

    # Morphological cleanup
    kernel_open  = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open,  iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)

    total_count, annotated = _count_from_mask(mask, annotated, ref_area, ref_shape=ref_shape)

    return total_count, annotated


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _white_pill_mask(blurred_bgr, mean_ref_value):
    """Binary mask for white/light pills using value-channel thresholding."""
    gray = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2GRAY)

    if mean_ref_value > 127:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_val = max(int(mean_ref_value * 0.75), 100)
        mask2 = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.bitwise_and(mask, mask2)
    else:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return mask


def _color_pill_mask(blurred_bgr, color_profiles):
    """
    Build a combined binary mask from a list of color profiles.
    Each achromatic cluster uses V-channel only; colored clusters use full HSV range.
    All cluster masks are OR'd together.
    """
    hsv = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2HSV)
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for profile in color_profiles:
        if profile["is_achromatic"]:
            v_low  = int(profile["hsv_lower"][2])
            v_high = int(profile["hsv_upper"][2])
            v_chan = hsv[:, :, 2]
            cluster_mask = np.where((v_chan >= v_low) & (v_chan <= v_high),
                                    np.uint8(255), np.uint8(0))
        else:
            lower = profile["hsv_lower"].astype(np.uint8)
            upper = profile["hsv_upper"].astype(np.uint8)
            if profile["wraps_hue"]:
                m1 = cv2.inRange(hsv,
                    np.array([lower[0], lower[1], lower[2]], dtype=np.uint8),
                    np.array([179,      upper[1], upper[2]], dtype=np.uint8))
                m2 = cv2.inRange(hsv,
                    np.array([0,        lower[1], lower[2]], dtype=np.uint8),
                    np.array([upper[0], upper[1], upper[2]], dtype=np.uint8))
                cluster_mask = cv2.bitwise_or(m1, m2)
            else:
                cluster_mask = cv2.inRange(hsv, lower, upper)

        combined = cv2.bitwise_or(combined, cluster_mask)

    return combined


def _compute_shape_metrics(contour):
    """Compute circularity, aspect_ratio, solidity for a contour."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)

    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / (hull_area + 1e-6)

    if len(contour) >= 5:
        e = cv2.fitEllipse(contour)
        ax1, ax2 = e[1]
        aspect_ratio = min(ax1, ax2) / (max(ax1, ax2) + 1e-6)
    else:
        rx, ry, rw, rh = cv2.boundingRect(contour)
        aspect_ratio = min(rw, rh) / (max(rw, rh) + 1e-6)

    return {
        "circularity":  float(circularity),
        "aspect_ratio": float(aspect_ratio),
        "solidity":     float(solidity),
    }


def _count_from_mask(mask, annotated, ref_area, ref_shape=None):
    """
    Count pills from a binary mask using watershed + area-ratio.
    Optionally validates detected regions against ref_shape.

    Returns (total_count, annotated_image).
    """
    h, w = mask.shape[:2]
    total_count = 0

    # Distance transform + watershed
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)

    _, sure_fg = cv2.threshold(dist_transform, 0.3, 1, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg * 255)

    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(mask, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    img_for_watershed = annotated.copy()
    markers = cv2.watershed(img_for_watershed, markers)
    annotated[markers == -1] = [0, 0, 255]  # red watershed boundaries

    # Analyse each watershed region
    pill_regions = []
    for label in np.unique(markers):
        if label <= 1:
            continue

        region_mask = np.uint8(markers == label) * 255
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        region_contour = max(contours, key=cv2.contourArea)
        region_area = cv2.contourArea(region_contour)

        if region_area < ref_area * 0.10:
            continue

        estimated_n = max(1, round(region_area / ref_area))
        pill_count = estimated_n

        # Shape validation
        if ref_shape is not None:
            metrics = _compute_shape_metrics(region_contour)
            scale = min(3.0, np.sqrt(estimated_n))
            tolerances = {
                "circularity":  0.35 * scale,
                "aspect_ratio": 0.30 * scale,
                "solidity":     0.25 * scale,
            }
            shape_ok = all(
                abs(metrics[k] - ref_shape[k]) <= tolerances[k]
                for k in tolerances
            )
            # Reject only tiny + wrong-shape regions (noise/debris)
            if not shape_ok and region_area < ref_area * 0.30:
                continue

        total_count += pill_count
        pill_regions.append((region_contour, region_area, pill_count))

    # Fallback: plain contour analysis if watershed gave nothing
    if total_count == 0:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < ref_area * 0.10:
                continue

            estimated_n = max(1, round(area / ref_area))

            if ref_shape is not None:
                metrics = _compute_shape_metrics(contour)
                scale = min(3.0, np.sqrt(estimated_n))
                tolerances = {
                    "circularity":  0.35 * scale,
                    "aspect_ratio": 0.30 * scale,
                    "solidity":     0.25 * scale,
                }
                shape_ok = all(
                    abs(metrics[k] - ref_shape[k]) <= tolerances[k]
                    for k in tolerances
                )
                if not shape_ok and area < ref_area * 0.30:
                    continue

            total_count += estimated_n
            pill_regions.append((contour, area, estimated_n))

    # Draw annotations
    colors = [
        (0, 255, 0),    # green
        (255, 165, 0),  # orange
        (0, 255, 255),  # yellow
        (255, 0, 255),  # magenta
        (0, 128, 255),  # blue-orange
    ]

    for i, (contour, area, pill_count) in enumerate(pill_regions):
        color = colors[i % len(colors)]
        cv2.drawContours(annotated, [contour], -1, color, 2)

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            rx, ry, rw, rh = cv2.boundingRect(contour)
            cx, cy = rx + rw // 2, ry + rh // 2

        label_text = str(pill_count)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(1.0, max(0.4, area / ref_area * 0.3))
        thickness = max(1, int(font_scale * 2))
        (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

        rx1 = max(0,     cx - tw // 2 - 4)
        ry1 = max(0,     cy - th // 2 - 4)
        rx2 = min(w - 1, cx + tw // 2 + 4)
        ry2 = min(h - 1, cy + th // 2 + 4)

        cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), color, -1)
        cv2.putText(annotated, label_text,
                    (cx - tw // 2, cy + th // 2),
                    font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # Total count overlay
    total_label = f"Total: {total_count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(total_label, font, 1.2, 2)
    cv2.rectangle(annotated, (10, 10), (tw + 20, th + 20), (0, 0, 0), -1)
    cv2.putText(annotated, total_label, (15, th + 13), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    return total_count, annotated
