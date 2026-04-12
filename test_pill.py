import cv2, numpy as np
from PIL import Image, ImageOps
import io

def load_image(path):
    with open(path, 'rb') as f:
        data = f.read()
    pil_image = Image.open(io.BytesIO(data))
    pil_image = ImageOps.exif_transpose(pil_image)
    pil_image = pil_image.convert('RGB')
    w, h = pil_image.size
    if max(w, h) > 2000:
        scale = 2000 / max(w, h)
        pil_image = pil_image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

from pill_counter import analyze_reference, count_pills

# ── Test set 1: white oval pills on blue tray ─────────────────────────────────
print('=== Test set 1: white oval pills on blue tray ===')
img1_ref = load_image('IMG_1241.JPG')
img1_grp = load_image('IMG_1242.JPG')

ref_area, color_profiles, is_white, ref_shape, bg_model = analyze_reference(img1_ref)
print('ref_area:', round(ref_area, 1), '| is_white:', is_white)

count1, ann1 = count_pills(img1_ref, ref_area, color_profiles,
                           is_white=is_white, ref_shape=ref_shape, bg_model=bg_model)
print('Single-pill self-test (IMG_1241):', count1, '(expected: 1)')
cv2.imwrite('result_white_single.jpg', ann1)

count2, ann2 = count_pills(img1_grp, ref_area, color_profiles,
                           is_white=is_white, ref_shape=ref_shape, bg_model=bg_model)
print('Group photo     (IMG_1242):', count2, '(expected: ~25)')
cv2.imwrite('result_white_group.jpg', ann2)

# ── Test set 2: orange round pills on gray tray ───────────────────────────────
print()
print('=== Test set 2: orange round pills on gray tray ===')
img2_ref = load_image('IMG_1243.jpeg')
img2_grp = load_image('IMG_1244.jpeg')

ref_area2, color_profiles2, is_white2, ref_shape2, bg_model2 = analyze_reference(img2_ref)
print('ref_area:', round(ref_area2, 1), '| is_white:', is_white2)

count3, ann3 = count_pills(img2_ref, ref_area2, color_profiles2,
                           is_white=is_white2, ref_shape=ref_shape2, bg_model=bg_model2)
print('Single-pill self-test (IMG_1243):', count3, '(expected: 1)')
cv2.imwrite('result_orange_single.jpg', ann3)

count4, ann4 = count_pills(img2_grp, ref_area2, color_profiles2,
                           is_white=is_white2, ref_shape=ref_shape2, bg_model=bg_model2)
print('Group photo     (IMG_1244):', count4, '(expected: 5)')
cv2.imwrite('result_orange_group.jpg', ann4)

print()
print('Done. Annotated results written to result_*.jpg')
