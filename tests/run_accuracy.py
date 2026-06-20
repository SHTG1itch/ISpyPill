"""
Accuracy harness: runs the pill counter over a matrix of synthetic cases with
known ground truth and reports error statistics.  Run:  python tests/run_accuracy.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pill_counter import analyze_reference, count_pills
from tests.synth import make_case, PILL_COLORS, TRAY_COLORS


def run(save_dir=None):
    cases = []
    seed = 0
    # A broad matrix: every pill colour on a couple of trays, all shapes, a
    # range of counts, separated and touching.
    pill_tray = [
        ("white", "blue"), ("white", "gray"), ("offwhite", "blue"),
        ("red", "white"), ("red", "gray"), ("orange", "gray"),
        ("yellow", "gray"), ("yellow", "black"), ("green", "white"),
        ("blue", "white"), ("blue", "gray"), ("pink", "gray"),
        ("brown", "white"), ("purple", "gray"), ("white", "black"),
        ("orange", "blue"),
    ]
    shapes = ["round", "oval", "capsule"]
    counts = [1, 4, 8, 15, 25]

    for (pc, tc) in pill_tray:
        for shape in shapes:
            for n in counts:
                seed += 1
                touching = (n >= 15)
                cases.append((pc, tc, shape, n, touching, seed))

    rows = []
    abs_err = []
    within1 = 0
    within2 = 0
    exact = 0
    for (pc, tc, shape, n, touching, seed) in cases:
        ref, grp, gt = make_case(seed, pc, tc, shape, n, touching)
        try:
            ra, ph, bh, ia, rs = analyze_reference(ref)
            pred, ann = count_pills(grp, ra, ph, bh, is_achromatic=ia, ref_shape=rs)
        except Exception as e:
            pred = -1
            ann = None
        err = abs(pred - gt) if pred >= 0 else 99
        abs_err.append(err)
        exact += (err == 0)
        within1 += (err <= 1)
        within2 += (err <= 2)
        rows.append((pc, tc, shape, gt, pred, err))
        if save_dir and ann is not None and err > 2:
            os.makedirs(save_dir, exist_ok=True)
            import cv2
            cv2.imwrite(os.path.join(save_dir, f"bad_{pc}_{tc}_{shape}_gt{gt}_pred{pred}.jpg"), ann)

    total = len(cases)
    print(f"{'pill':9} {'tray':7} {'shape':8} {'gt':>3} {'pred':>4} {'err':>3}")
    print("-" * 42)
    for r in rows:
        flag = "" if r[5] <= 1 else ("  <-" if r[5] <= 2 else "  <<<<")
        print(f"{r[0]:9} {r[1]:7} {r[2]:8} {r[3]:>3} {r[4]:>4} {r[5]:>3}{flag}")
    print("-" * 42)
    print(f"cases={total}  exact={exact} ({100*exact/total:.0f}%)  "
          f"within1={within1} ({100*within1/total:.0f}%)  "
          f"within2={within2} ({100*within2/total:.0f}%)  "
          f"MAE={np.mean(abs_err):.2f}")
    return rows, abs_err


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Synthetic pill-count accuracy harness")
    ap.add_argument("--save-bad", metavar="DIR", default=None,
                    help="write annotated images for cases off by >2 to DIR")
    run(save_dir=ap.parse_args().save_bad)
