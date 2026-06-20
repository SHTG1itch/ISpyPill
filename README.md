# ISpyPill 💊 — Count pills from a photo

ISpyPill counts pills from two photos. You take **one picture of a single pill**
(the reference) and **one picture of a pile or spread of pills** (the group), and
the app tells you how many pills are in the group and draws an annotated image
showing what it counted.

It works for **any pill colour and shape** — white tablets, orange rounds, red
caplets, two‑tone capsules — by learning what *your* pill and *your* tray look
like, then finding the pills that sit on that tray. Everything runs **locally on
your machine**; your photos are never uploaded anywhere.

| | |
|---|---|
| 🎯 **Use case** | Medication management, inventory, accessibility |
| 🖥️ **Runs on** | Web browser (desktop & mobile) and a native iOS/Android app |
| 🔒 **Privacy** | 100% local — no cloud, no accounts |
| ⚡ **Speed** | ~1–2 seconds per photo |

---

## 📑 Table of contents

- [How it works (in one minute)](#-how-it-works-in-one-minute)
- [Tech stack](#-tech-stack)
- [Quick start](#-quick-start)
- [Using the app](#-using-the-app)
- [Tips for accurate counts](#-tips-for-accurate-counts)
- [The counting algorithm](#-the-counting-algorithm)
- [API reference](#-api-reference)
- [Testing & accuracy](#-testing--accuracy)
- [Project structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Limitations](#-limitations)
- [License](#-license)

---

## 🔍 How it works (in one minute)

1. **Reference photo** — one pill, centred, on the tray/surface you'll use.
   The app learns the pill's size, colour and shape **and** the tray's colour.
2. **Group photo** — many pills of the same type spread on that tray.
3. **Count** — the app isolates the tray, treats the pills as the objects
   sitting on it, separates touching pills, and returns the count plus an
   annotated image.

The key insight: *a pill is an object resting **on** the reference surface.*
This is why it works even when the pills are the same colour as something else in
the frame (e.g. white pills on a blue tray standing on a white counter — the
counter is ignored because it isn't part of the tray).

---

## 🧰 Tech stack

| Layer | Technology |
|-------|-----------|
| **Backend / computer vision** | Python 3.10+ · [Flask](https://flask.palletsprojects.com/) · [OpenCV](https://opencv.org/) (`opencv-python-headless`) · [NumPy](https://numpy.org/) · [SciPy](https://scipy.org/) · [Pillow](https://python-pillow.org/) |
| **Web app** | [React](https://react.dev/) 18 + [Vite](https://vitejs.dev/) 5 |
| **Mobile app** | [Expo](https://expo.dev/) (SDK 54) / [React Native](https://reactnative.dev/) 0.81, `expo-image-picker` |
| **Tests** | [pytest](https://pytest.org/) + a synthetic accuracy harness |

The Flask backend exposes a single `/analyze` endpoint and also serves the built
React app, so in production you only need to run **one** server.

---

## 🚀 Quick start

### Prerequisites

| Tool | Version | Needed for |
|------|---------|-----------|
| **Python** | 3.10+ | Backend / counting (required) |
| **Node.js + npm** | Node 18+ | Web UI and/or mobile UI |

Check what you have:

```bash
python --version        # 3.10 or higher
node --version          # 18 or higher
```

> On macOS/Linux you may need `python3` / `pip3` instead of `python` / `pip`.

### Step 1 — Backend (required)

```bash
# from the repo root
pip install -r requirements.txt
python app.py
```

You should see `Running on http://127.0.0.1:5000`. The backend is now ready.

### Step 2 — Web app

Pick **one** of the two ways to run the UI.

**Option A — Development mode (hot reload, recommended while editing):**
Open two terminals.

```bash
# Terminal 1 — backend
python app.py                       # http://localhost:5000

# Terminal 2 — frontend dev server
cd frontend
npm install                         # first time only (~1–3 min)
npm run dev                         # http://localhost:5173
```

Open **http://localhost:5173**. The Vite dev server automatically proxies API
calls (`/analyze`) to Flask on port 5000, so both pieces work together.

**Option B — Production mode (single server, no Node needed at runtime):**

```bash
cd frontend
npm install                         # first time only
npm run build                       # creates frontend/dist/
cd ..
python app.py
```

Open **http://localhost:5000** — Flask serves the built React app *and* the API.

### Step 3 — Mobile app (optional)

A native iOS/Android client built with Expo lives in `mobile/`.

```bash
cd mobile
npm install
npm start                           # or: npm run ios / npm run android
```

Scan the QR code with **Expo Go** on your phone. Your phone and computer must be
on the **same Wi‑Fi network**, and the Flask backend (Step 1) must be running —
the app auto‑detects your computer's LAN address and calls `http://<that-ip>:5000`.

---

## 📱 Using the app

1. **Add a reference pill** — upload or photograph **one** pill, centred and
   well‑lit, on the surface you'll lay the rest out on.
2. **Add a group photo** — upload or photograph the pills you want counted,
   spread out (ideally in a single layer) on that same surface.
3. **Count** — press the button; results appear in 1–2 seconds.
4. **Save / retry** — download the annotated image, or reset and count again.

---

## 💡 Tips for accurate counts

- **Use a contrasting surface.** Dark pills on a light tray, light pills on a
  dark/coloured tray. (A coloured tray such as a blue pill‑tray works great even
  for white pills.)
- **Spread the pills out** in a single layer. Touching is fine; **stacking /
  piling is not** — overlapping pills can't all be seen.
- **Even lighting, in focus.** Glare is handled, but avoid deep shadows and blur.
- **Same surface for both photos** — the reference teaches the app what the tray
  looks like, so use the same tray for the group shot.
- **Shoot from above** (top‑down), not a steep angle.

---

## 🔬 The counting algorithm

ISpyPill is built on the idea that **a pill is an object resting on the reference
surface (the tray)** — so instead of segmenting pills by their own colour across
the whole image (which fails when the surroundings share that colour), it learns
both the pill *and* the tray, isolates the tray, and counts what sits on it.

### 1. Reference analysis (`analyze_reference`)
- **Isolates the pill** with a scored ensemble of cues (saturation Otsu,
  brightness Otsu, and distance‑from‑background‑colour), choosing the contour
  that is most solid, circular, centred and pill‑sized — robust without relying
  on a single threshold.
- **Learns pill vs. tray appearance** by sampling the pill interior and an
  annulus of the tray *immediately around it* (not the whole frame, which would
  pull in distant same‑coloured material). It records median **saturation**,
  **value (brightness)** and **hue** for both, plus Hue‑Saturation histograms.
- **Measures shape**: circularity, aspect ratio, solidity.

### 2. Counting (`count_pills`)
- **Picks the most discriminative channel** from the reference statistics:

  | Cue | Best for | Why |
  |-----|----------|-----|
  | **Saturation** | pale pill on a coloured tray (white on blue) | Invariant to a glossy tray's glare/brightness gradient — preferred when it separates |
  | **Hue** | two saturated colours (orange on blue) | Per‑pixel nearest‑hue is invariant to shading |
  | **Value** | achromatic on achromatic (white on grey/steel, orange on dark wood) | Flat‑field correction removes illumination & glare before thresholding |

- **Builds the tray ROI** — the dominant tray region with its holes filled — so
  same‑coloured material *outside* the tray is excluded.
- **Re‑estimates the single‑pill size** from genuinely isolated pills in the
  group photo, which auto‑corrects an inaccurate reference area and adapts to
  camera distance.
- **Counts** each blob: single‑pill‑sized blobs are shape‑validated and count as
  one; touching clusters are counted by area ratio (total area ÷ pill area).

### 3. Result
Draws coloured outlines with per‑region counts, overlays the total, and returns
the annotated image as base64 JPEG.

The two public functions live in `pill_counter.py`:

```python
ref_area, pill_hist, bg_hist, is_achromatic, ref_shape = analyze_reference(ref_bgr)
count, annotated_bgr = count_pills(group_bgr, ref_area, pill_hist, bg_hist,
                                   is_achromatic=is_achromatic, ref_shape=ref_shape)
```

---

## 🔌 API reference

The backend (default `http://localhost:5000`) exposes:

### `POST /analyze`
Counts pills. Send a `multipart/form-data` body with two image files:

| Field | Description |
|-------|-------------|
| `reference_pill` | photo of one pill |
| `group_photo` | photo of the pills to count |

**Success `200`:**
```json
{
  "count": 12,
  "annotated_image": "<base64 JPEG>",
  "ref_area_px": 10840.0,
  "is_white_pill": false,
  "num_color_clusters": 1
}
```

**Errors:** `400` (missing file), `422` (couldn't isolate the reference pill —
usually a low‑contrast photo), `500` (unexpected processing error).

Example:
```bash
curl -X POST http://localhost:5000/analyze \
  -F "reference_pill=@IMG_1241.JPG" \
  -F "group_photo=@IMG_1242.JPG"
```

### `GET /ping`
Health check used by the mobile app: returns `{"status": "ok"}`.

> Uploads are capped at **16 MB** and the longest image side is resized to
> **2000 px** for speed. EXIF orientation is corrected automatically.

---

## 🧪 Testing & accuracy

```bash
# Unit tests + real-image tests (fast, deterministic)
pytest test_unit.py tests/test_online.py

# Self-test on the four bundled phone photos (white pills on blue tray,
# orange pills on grey tray) — prints counts vs. expected
python test_pill.py

# Synthetic accuracy harness: counts thousands of generated pills across
# many colours × shapes × counts (exact known ground truth) and reports
# exact / within-1 / within-2 rates and mean absolute error
python tests/run_accuracy.py
```

**Current accuracy** on the synthetic matrix (10 pill colours × 6 tray colours ×
3 shapes × 5 counts, 240 cases): **~99.6% exact, mean error 0.02 pills.** The
four bundled real photos are counted accurately (single‑pill self‑tests = 1,
orange group exact, white group within ±1), as is the spread‑out real test image
in `tests/online_images/` (6 pills, counted exactly).

Test assets:
- `IMG_1241`–`IMG_1244` — real phone photos used by `test_pill.py`.
- `tests/online_images/` — freely‑licensed photos from Wikimedia Commons
  (see `tests/online_images/SOURCES.md`).
- `tests/synth.py` — the synthetic pill‑image generator used by the harness.

---

## 🗂️ Project structure

```
ISpyPill/
├── app.py                      # Flask server: /analyze, /ping, serves the React app
├── pill_counter.py             # Core computer-vision pipeline (the algorithm)
├── requirements.txt            # Python dependencies
│
├── frontend/                   # React + Vite web app (the active UI)
│   ├── package.json            #   scripts: dev / build / preview
│   ├── vite.config.js          #   dev server :5173, proxies /analyze → :5000
│   └── src/
│       ├── App.jsx             #   main component & upload/count logic
│       ├── App.css
│       └── components/         #   UploadCard, ResultsCard, ErrorBanner
│
├── mobile/                     # Expo / React Native app (iOS & Android)
│   ├── App.js
│   └── config.js               #   auto-resolves the Flask LAN URL
│
├── test_pill.py                # Quick self-test on the bundled photos
├── test_unit.py                # pytest unit tests
└── tests/
    ├── synth.py                # synthetic pill-image generator
    ├── run_accuracy.py         # accuracy harness
    ├── test_online.py          # real-image tests
    └── online_images/          # downloaded real test photos + SOURCES.md
```

> **Note:** `static/` and `templates/` contain an older vanilla‑JavaScript UI
> that has been **superseded by the React app in `frontend/`** (Flask serves
> `frontend/dist`, not these). They are kept for reference only.

---

## 🛠️ Troubleshooting

**"React build not found" when running `python app.py`**
Build the frontend first: `cd frontend && npm install && npm run build && cd ..`.
(Or use dev mode — Option A above — and open port 5173 instead.)

**`ModuleNotFoundError: No module named 'flask'` (or cv2 / scipy)**
Install the Python deps: `pip install -r requirements.txt`.

**Port 5000 already in use**
Stop the other process, or change the port at the bottom of `app.py`
(`app.run(..., port=5000)`).

**The web app loads but counting fails with 422**
The reference photo couldn't be isolated — retake it with the pill centred and
clearly contrasting against the background.

**Mobile app can't reach the backend**
Phone and computer must be on the same Wi‑Fi, the Flask server must be running,
and your firewall must allow inbound connections to port 5000.

**Counts look off**
See [Tips for accurate counts](#-tips-for-accurate-counts) — most issues are
lighting, low contrast, or stacked/overlapping pills.

---

## ⚠️ Limitations

- **Single layer only.** The app counts what it can see; pills stacked in a 3‑D
  pile can't all be seen, so dense heaps are under/over‑estimated.
- **Needs contrast.** Pills on a same‑coloured, textured surface with no tray
  (e.g. white capsules on white paper) are hard to separate — use a contrasting
  tray.
- **One pill type per count.** The reference defines the pill being counted;
  mixing several different pills in the group photo isn't supported.

---

## 📝 License

Licensed under the **MIT License** — use, modify and distribute freely,
including commercially, provided the original license is included.

---

**Happy counting! 💊**
