"""
Synthetic pill-image generator for accuracy testing.

Produces realistic (ref, group, ground_truth) triples where the count is known
exactly.  Covers many pill colours, tray colours, shapes (round / oval /
capsule), counts and packing densities — the cases real photos cannot pin down
to an exact integer.  Images include lighting gradients, glare highlights,
embossing and sensor noise so the segmentation is genuinely exercised.
"""

import cv2
import numpy as np


# BGR colours
PILL_COLORS = {
    "white":  (238, 238, 240),
    "offwhite": (225, 230, 238),
    "red":    (40, 40, 200),
    "orange": (40, 110, 225),
    "yellow": (60, 210, 235),
    "green":  (90, 180, 90),
    "blue":   (200, 120, 40),
    "pink":   (170, 150, 240),
    "brown":  (60, 80, 120),
    "purple": (160, 70, 130),
}

TRAY_COLORS = {
    "blue":   (190, 120, 40),
    "gray":   (135, 135, 135),
    "white":  (235, 235, 238),
    "black":  (35, 35, 35),
    "green":  (90, 150, 70),
    "wood":   (70, 120, 165),
}


def _rng(seed):
    return np.random.RandomState(seed)


def _draw_tray(canvas, tray_bgr, rng):
    """Fill a tray region with colour + lighting gradient + noise + glare."""
    h, w = canvas.shape[:2]
    canvas[:] = tray_bgr
    # smooth lighting gradient
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    grad = (np.cos(ang) * gx + np.sin(ang) * gy)
    grad = (grad - grad.min()) / (np.ptp(grad) + 1e-6)
    factor = 0.75 + 0.5 * grad           # 0.75 .. 1.25
    canvas[:] = np.clip(canvas.astype(np.float32) * factor[..., None], 0, 255).astype(np.uint8)
    # a soft glare blob (kept moderate: a real glossy tray's sheen brightens the
    # surface but does not blow it out to the pill's brightness)
    if rng.rand() < 0.7:
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        r = rng.randint(min(h, w) // 6, min(h, w) // 3)
        glare = np.zeros((h, w), np.float32)
        cv2.circle(glare, (cx, cy), r, 1.0, -1)
        glare = cv2.GaussianBlur(glare, (0, 0), r / 2)
        canvas[:] = np.clip(canvas.astype(np.float32) +
                            glare[..., None] * rng.uniform(22, 48), 0, 255).astype(np.uint8)
    # sensor noise
    noise = rng.normal(0, 4, canvas.shape).astype(np.float32)
    canvas[:] = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _pill_axes(shape, base):
    if shape == "round":
        return base, base
    if shape == "oval":
        return int(base * 1.5), int(base * 0.9)
    return int(base * 2.1), int(base * 0.8)   # capsule


def _draw_pill(canvas, center, axes, angle, pill_bgr, rng):
    """Draw one pill with shading, an embossed line and a glare streak."""
    h, w = canvas.shape[:2]
    overlay = canvas.copy()
    a, b = axes
    cv2.ellipse(overlay, center, (a, b), angle, 0, 360, pill_bgr, -1, cv2.LINE_AA)
    # subtle 3-D shading: darker rim
    rim = tuple(int(c * 0.82) for c in pill_bgr)
    cv2.ellipse(overlay, center, (a, b), angle, 0, 360, rim, max(2, b // 6), cv2.LINE_AA)
    # embossed score line (darker)
    emb = tuple(int(c * 0.7) for c in pill_bgr)
    rad = np.deg2rad(angle)
    dx, dy = np.cos(rad + np.pi / 2), np.sin(rad + np.pi / 2)
    p1 = (int(center[0] - dx * b * 0.8), int(center[1] - dy * b * 0.8))
    p2 = (int(center[0] + dx * b * 0.8), int(center[1] + dy * b * 0.8))
    cv2.line(overlay, p1, p2, emb, max(1, b // 8), cv2.LINE_AA)
    # glare highlight (lighter), offset
    hi = tuple(min(255, int(c * 1.18 + 25)) for c in pill_bgr)
    hcx = int(center[0] - np.cos(rad) * a * 0.25)
    hcy = int(center[1] - np.sin(rad) * a * 0.25)
    cv2.ellipse(overlay, (hcx, hcy), (max(2, a // 4), max(1, b // 4)),
                angle, 0, 360, hi, -1, cv2.LINE_AA)
    canvas[:] = cv2.addWeighted(overlay, 0.95, canvas, 0.05, 0)


def _place(n, h, w, axes, rng, touching):
    """Return n non-overlapping (or lightly touching) centres+angles."""
    a, b = axes
    reach = a + 6
    margin = reach + 6
    centers = []
    tries = 0
    # Flat pills on a tray touch edge-to-edge but do not stack/overlap; a tiny
    # negative gap models just-touching, a positive gap models spread out.
    gap = -0.04 if touching else 0.12
    while len(centers) < n and tries < n * 400:
        tries += 1
        cx = rng.randint(margin, w - margin)
        cy = rng.randint(margin, h - margin)
        ang = rng.uniform(0, 180)
        ok = True
        for (px, py, _) in centers:
            if np.hypot(cx - px, cy - py) < 2 * reach * (1 + gap):
                ok = False
                break
        if ok:
            centers.append((cx, cy, ang))
    return centers


def make_case(seed, pill_color, tray_color, shape, n, touching=False,
              group_size=(900, 1200), ref_size=(500, 650)):
    """Return (ref_bgr, group_bgr, ground_truth_n)."""
    rng = _rng(seed)
    pill_bgr = PILL_COLORS[pill_color]
    tray_bgr = TRAY_COLORS[tray_color]

    # group image
    gh, gw = group_size
    group = np.zeros((gh, gw, 3), np.uint8)
    _draw_tray(group, tray_bgr, rng)
    base = rng.randint(34, 52)
    axes = _pill_axes(shape, base)
    centers = _place(n, gh, gw, axes, rng, touching)
    for (cx, cy, ang) in centers:
        jitter = tuple(int(np.clip(c + rng.normal(0, 6), 0, 255)) for c in pill_bgr)
        _draw_pill(group, (cx, cy), axes, ang, jitter, rng)
    actual = len(centers)        # placement may fall short for dense cases

    # reference image: one pill, centred, same look
    rh, rw = ref_size
    ref = np.zeros((rh, rw, 3), np.uint8)
    _draw_tray(ref, tray_bgr, _rng(seed + 7))
    rbase = int(base * 1.7)
    raxes = _pill_axes(shape, rbase)
    _draw_pill(ref, (rw // 2, rh // 2), raxes, rng.uniform(0, 180), pill_bgr, rng)

    return ref, group, actual
