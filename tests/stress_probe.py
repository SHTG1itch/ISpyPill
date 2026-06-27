"""
Exploratory stress probe — NOT a fixture. Adds realistic conditions the main
synthetic generator omits (cast shadows, border-cropped pills, low pill/tray
contrast, dense touching at low counts) to surface where the counter miscounts.
Run: python tests/stress_probe.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
from pill_counter import analyze_reference, count_pills
from tests.synth import (PILL_COLORS, TRAY_COLORS, _draw_tray, _pill_axes,
                         _draw_pill, _place, _rng)


def _draw_shadow(canvas, center, axes, angle, rng):
    """Soft dark cast shadow offset down-right under a pill."""
    h, w = canvas.shape[:2]
    a, b = axes
    off = max(4, a // 5)
    shadow = np.zeros((h, w), np.float32)
    cv2.ellipse(shadow, (center[0] + off, center[1] + off),
                (int(a * 1.05), int(b * 1.05)), angle, 0, 360, 1.0, -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), max(3, a // 3))
    darken = 1.0 - shadow[..., None] * rng.uniform(0.25, 0.45)
    canvas[:] = np.clip(canvas.astype(np.float32) * darken, 0, 255).astype(np.uint8)


def make_case_stress(seed, pill_color, tray_color, shape, n, touching,
                     shadows=False, border=False,
                     group_size=(900, 1200), ref_size=(500, 650)):
    rng = _rng(seed)
    pill_bgr = PILL_COLORS[pill_color]
    tray_bgr = TRAY_COLORS[tray_color]
    gh, gw = group_size
    group = np.zeros((gh, gw, 3), np.uint8)
    _draw_tray(group, tray_bgr, rng)
    base = rng.randint(34, 52)
    axes = _pill_axes(shape, base)
    centers = _place(n, gh, gw, axes, rng, touching)
    if border and centers:
        # Shove a couple of pills against the frame edge (partially cropped).
        centers[0] = (axes[0] // 2, centers[0][1], centers[0][2])
        if len(centers) > 1:
            centers[1] = (centers[1][0], gh - axes[1] // 2, centers[1][2])
    if shadows:
        for (cx, cy, ang) in centers:
            _draw_shadow(group, (cx, cy), axes, ang, rng)
    for (cx, cy, ang) in centers:
        jitter = tuple(int(np.clip(c + rng.normal(0, 6), 0, 255)) for c in pill_bgr)
        _draw_pill(group, (cx, cy), axes, ang, jitter, rng)
    actual = len(centers)

    rh, rw = ref_size
    ref = np.zeros((rh, rw, 3), np.uint8)
    _draw_tray(ref, tray_bgr, _rng(seed + 7))
    raxes = _pill_axes(shape, int(base * 1.7))
    _draw_pill(ref, (rw // 2, rh // 2), raxes, rng.uniform(0, 180), pill_bgr, rng)
    return ref, group, actual


def run_suite(name, cases, **kw):
    abs_err, exact, within1, bad = [], 0, 0, []
    for (pc, tc, shape, n, touching, seed) in cases:
        ref, grp, gt = make_case_stress(seed, pc, tc, shape, n, touching, **kw)
        try:
            ra, ph, bh, ia, rs = analyze_reference(ref)
            pred, _ = count_pills(grp, ra, ph, bh, is_achromatic=ia, ref_shape=rs)
        except Exception:
            pred = -1
        err = abs(pred - gt) if pred >= 0 else 99
        abs_err.append(err)
        exact += (err == 0)
        within1 += (err <= 1)
        if err > 1:
            bad.append((pc, tc, shape, gt, pred, err))
    t = len(cases)
    print(f"[{name:16}] cases={t}  exact={100*exact/t:.0f}%  "
          f"within1={100*within1/t:.0f}%  MAE={np.mean(abs_err):.2f}")
    for b in bad[:25]:
        print(f"      MISS {b[0]:8} {b[1]:6} {b[2]:7} gt={b[3]:>3} pred={b[4]:>3} err={b[5]}")
    return abs_err


def build_cases():
    pill_tray = [
        ("white", "blue"), ("white", "gray"), ("red", "gray"),
        ("orange", "gray"), ("yellow", "black"), ("green", "white"),
        ("blue", "white"), ("white", "black"), ("orange", "blue"),
    ]
    shapes = ["round", "oval", "capsule"]
    counts = [4, 8, 15, 25]
    cases, seed = [], 1000
    for (pc, tc) in pill_tray:
        for shape in shapes:
            for n in counts:
                seed += 1
                cases.append((pc, tc, shape, n, n >= 12, seed))
    return cases


if __name__ == "__main__":
    cases = build_cases()
    run_suite("baseline", cases)
    run_suite("shadows", cases, shadows=True)
    run_suite("border-crop", cases, border=True)
    run_suite("shadows+border", cases, shadows=True, border=True)
    # dense touching at LOW counts (n=4,8 forced touching)
    dense = [(pc, tc, sh, n, True, sd) for (pc, tc, sh, n, _, sd) in cases if n in (4, 8)]
    run_suite("dense-low-touch", dense)
