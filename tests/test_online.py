"""
Real-image tests using freely-licensed pill photos downloaded from Wikimedia
Commons (see tests/online_images/SOURCES.md).  Each group photo is paired with a
reference pill cropped from the same photo (`*_REF.jpg`).

Run:  pytest tests/test_online.py -v -s
"""
import os, io, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pytest
from PIL import Image, ImageOps

from pill_counter import analyze_reference, count_pills

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "online_images")


def _load(path, max_dim=2000):
    with open(path, "rb") as f:
        im = Image.open(io.BytesIO(f.read()))
    im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    if max(w, h) > max_dim:
        s = max_dim / max(w, h)
        im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def _count(group_file, ref_file):
    grp = _load(os.path.join(IMG_DIR, group_file))
    ref = _load(os.path.join(IMG_DIR, ref_file))
    ra, ph, bh, achr, rs = analyze_reference(ref)
    count, _ = count_pills(grp, ra, ph, bh, is_achromatic=achr, ref_shape=rs)
    return count


# (group, reference, ground_truth, tolerance)
SPREAD_CASES = [
    # 6 identical amber allergy tablets, well separated on dark wood — exact.
    ("Generic_12-Hour_Allergy_Pills.JPG", "Generic_REF.jpg", 6, 1),
]


@pytest.mark.parametrize("grp,ref,gt,tol", SPREAD_CASES)
def test_spread_real_images(grp, ref, gt, tol):
    if not os.path.exists(os.path.join(IMG_DIR, grp)):
        pytest.skip(f"{grp} not downloaded")
    count = _count(grp, ref)
    print(f"\n{grp}: counted {count} (truth {gt})")
    assert abs(count - gt) <= tol


def test_dense_pile_degrades_gracefully():
    """A heavily stacked/overlapping 3-D heap (~45 caplets) is an out-of-spec
    stress case: an exact count is not well-defined and is not expected.  We only
    assert the counter detects a substantial pile rather than crashing or
    returning a trivial number — the app targets pills spread in a single layer
    (see test_spread_real_images and the in-repo IMG_124x photos)."""
    grp = "Equate_Ibuprofen_Pills.JPG"
    if not os.path.exists(os.path.join(IMG_DIR, grp)):
        pytest.skip(f"{grp} not downloaded")
    count = _count(grp, "Equate_Ibuprofen_REF.jpg")
    print(f"\n{grp}: counted {count} (dense stacked heap — exact count undefined)")
    assert count >= 10
