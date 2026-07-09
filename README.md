# ISpyPill — Count pills from a photo

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
| **Use case** | Medication management, inventory, accessibility |
| **Runs on** | Web browser (desktop & mobile) and a native iOS/Android app |
| **Privacy** | 100% local — no cloud, no accounts |
| **Speed** | ~1–2 s for typical phone photos, up to ~10 s at the 2000 px cap |

---

## Table of contents

- [How it works (in one minute)](#how-it-works-in-one-minute)
- [Tech stack](#tech-stack)
- [Quick start](#quick-start)
- [Using the app](#using-the-app)
- [Tips for accurate counts](#tips-for-accurate-counts)
- [Common pitfalls](#common-pitfalls)
- [The counting algorithm](#the-counting-algorithm)
- [How the app avoids miscounting](#how-the-app-avoids-miscounting)
- [API reference](#api-reference)
- [Testing & accuracy](#testing--accuracy)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [FAQ](#faq)
- [License](#license)

---

## How it works (in one minute)

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

**Why two photos?** The reference photo is what makes the app work for *any* pill
without being pre-trained on specific medications. Instead of guessing what a
"pill" looks like, the app measures **your** exact pill (its size, colour, hue,
brightness and shape) and **your** exact tray, then looks for that combination in
the group photo. Change the medication or the tray and you just take a new
reference photo — there is no model to retrain and nothing is hard-coded to a
particular drug.

**What "counting" actually does**, step by step:

1. Find the tray (the dominant surface the reference pill was sitting on).
2. Treat every object resting on that tray as a pill — regardless of its colour.
3. Work out the size of a single pill *from the group photo itself*, so the count
 stays correct even if the group photo was taken closer or farther than the
 reference.
4. Count isolated pills directly, and estimate touching clusters by area
 (cluster area ÷ one-pill area).
5. Draw the outlines and numbers onto a copy of your photo so you can **see and
 double-check** exactly what was counted.

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| **Backend / computer vision** | Python 3.10+ · [Flask](https://flask.palletsprojects.com/) · [OpenCV](https://opencv.org/) (`opencv-python-headless`) · [NumPy](https://numpy.org/) · [SciPy](https://scipy.org/) · [Pillow](https://python-pillow.org/) |
| **Web app** | [React](https://react.dev/) 18 + [Vite](https://vitejs.dev/) 5 |
| **Mobile app** | [Expo](https://expo.dev/) (SDK 54) / [React Native](https://reactnative.dev/) 0.81, `expo-image-picker` |
| **Tests** | [pytest](https://pytest.org/) + a synthetic accuracy harness |

The Flask backend exposes a single `/analyze` endpoint and also serves the built
React app, so in production you only need to run **one** server.

---

## Quick start

### Prerequisites

| Tool | Version | Needed for |
|------|---------|-----------|
| **Python** | 3.10+ | Backend / counting (required) |
| **Node.js + npm** | Node 18+ | Web UI and/or mobile UI |

Check what you have:

```bash
python --version # 3.10 or higher
node --version # 18 or higher
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
python app.py # http://localhost:5000

# Terminal 2 — frontend dev server
cd frontend
npm install # first time only (~1–3 min)
npm run dev # http://localhost:5173
```

Open **http://localhost:5173**. The Vite dev server automatically proxies API
calls (`/analyze`) to Flask on port 5000, so both pieces work together.

**Option B — Production mode (single server, no Node needed at runtime):**

```bash
cd frontend
npm install # first time only
npm run build # creates frontend/dist/
cd ..
python app.py
```

Open **http://localhost:5000** — Flask serves the built React app *and* the API.

### Step 3 — Run on your phone with Expo (optional)

A native iOS/Android client built with Expo lives in `mobile/`. You run it on a
real phone using the free **Expo Go** app — no Xcode or Android Studio required.

**One‑time setup on your phone:** install **Expo Go** from the App Store (iOS) or
Play Store (Android). Make sure it is up to date — this project targets Expo
SDK 54, and an outdated Expo Go will refuse to open the project.

**On your computer**, keep the Flask backend from Step 1 running in its own
terminal (`python app.py` — it must say *"Running on http://0.0.0.0:5000"*), then
start the Expo dev server in a second terminal. There are two ways to connect,
depending on your network.

#### Option A (default) — same Wi‑Fi (LAN)

> **Both devices must be on the same internet/Wi‑Fi network.** Your phone and the
> computer running the backend have to be connected to the **same Wi‑Fi** (not a
> "guest" network, and not the phone on cellular) — otherwise the phone can't
> reach the dev server or the backend.

```bash
cd mobile
npm install   # first time only — installs Expo + native modules
npx expo start   # starts Metro and prints a QR code
```

The app auto‑detects your computer's LAN address from the Expo dev server and
calls the backend at `http://<your-computer-ip>:5000` — no IP to hard‑code. Then
scan the QR code with your phone (see **Connect the phone** below).

#### Option B — tunnel (different networks, or restrictive Wi‑Fi)

Use this when LAN doesn't work — e.g. the phone is on cellular, you're on a
"guest"/corporate Wi‑Fi that blocks device‑to‑device traffic, or the QR code
loads forever. The tunnel routes Metro over the internet via ngrok.

> **Important:** the tunnel only forwards Metro (the JavaScript bundle) — it does
> **not** forward the Flask backend on port `5000`. So in tunnel mode the app
> cannot auto‑detect the backend; you must tell it where Flask is with the
> `EXPO_PUBLIC_API_URL` environment variable. Point it at a URL your phone can
> actually reach:
> - On the **same Wi‑Fi** (using tunnel only because LAN QR is flaky): use your
>   computer's LAN IP, e.g. `http://192.168.1.50:5000`.
> - On a **different network**: expose Flask publicly first (e.g. a separate
>   `ngrok http 5000`) and use that public URL.

```bash
cd mobile
npm install                                    # first time only

# PowerShell:
$env:EXPO_PUBLIC_API_URL = "http://192.168.1.50:5000"; npm run tunnel

# macOS / Linux / Git Bash:
EXPO_PUBLIC_API_URL=http://192.168.1.50:5000 npm run tunnel
```

`npm run tunnel` is shorthand for `npx expo start --tunnel`. The required ngrok
helper (`@expo/ngrok`) is already a dev dependency, so the tunnel starts without
prompting to install anything. (Run `python -c "import socket; print(socket.gethostbyname(socket.gethostname()))"`
or `ipconfig` to find your computer's LAN IP.)

**Connect the phone (either option):**

- **Android** — open Expo Go, tap **Scan QR code**, and scan the QR code in your
 terminal.
- **iOS** — open the **Camera** app, point it at the QR code, and tap the
 **Open in Expo Go** banner.

> **Windows firewall:** the first time Flask accepts a connection, Windows may pop
> up a "Windows Defender Firewall" dialog. Click **Allow access** (Private
> networks) so your phone can reach port `5000`. If you dismissed it and the phone
> shows "Server unreachable", allow Python through the firewall for private
> networks, or temporarily allow inbound TCP port `5000`.

**Sanity checks if the phone can't reach the server:**

- The header in the app shows a status dot — green "Server connected" means the
 phone reached Flask; red "Server unreachable" means it could not.
- From the phone's browser, open `http://<your-computer-ip>:5000/ping` — it should
 return `{"status": "ok"}`. If that fails, it's a network/firewall issue, not the
 app.
- Confirm both devices are on the same Wi‑Fi (not a "guest" network, and not the
 phone on cellular).

> **Dependency note:** the Expo app uses `@expo/vector-icons`, which needs
> `expo-font` and SDK‑matched versions of every native module. These are already
> pinned correctly in `mobile/package.json`; `npm install` is all you need. If you
> ever change versions, run `npx expo install --check` to keep them aligned and
> `npx expo-doctor` to verify (it should report **18/18 checks passed**).

---

## Using the app

1. **Add a reference pill** — upload or photograph **one** pill, centred and
 well‑lit, on the surface you'll lay the rest out on.
2. **Add a group photo** — upload or photograph the pills you want counted,
 spread out (ideally in a single layer) on that same surface.
3. **Count** — press the button; results appear in 1–2 seconds.
4. **Save / retry** — download the annotated image, or reset and count again.

---

## Tips for accurate counts

- **Use a contrasting surface.** Dark pills on a light tray, light pills on a
 dark/coloured tray. (A coloured tray such as a blue pill‑tray works great even
 for white pills.)
- **Spread the pills out** in a single layer. Touching is fine; **stacking /
 piling is not** — overlapping pills can't all be seen.
- **Even lighting, in focus.** Glare is handled, but avoid deep shadows and blur.
- **Same surface for both photos** — the reference teaches the app what the tray
 looks like, so use the same tray for the group shot.
- **Shoot from above** (top‑down), not a steep angle.
- **Keep all pills inside the frame.** The app can recover pills that touch the
 edge of the photo *when the tray fills the frame*, but a pill cut in half by the
 frame on a busy background is easy to miss — leave a small margin of tray
 around every pill.

---

## Common pitfalls

These are the mistakes that most often produce a wrong count, what they do, and
how to avoid them. (Always glance at the annotated image — it shows you exactly
what was and wasn't counted.)

| Pitfall | What happens | Fix |
|---------|--------------|-----|
| **Stacked / overlapping pills** | Pills hidden under others can't be seen -> **undercount** | Spread pills into a single layer; touching is fine, piling is not |
| **No tray / same-colour surface** | White pills on white paper have nothing to separate them from -> unreliable mask | Use a **contrasting tray** (a coloured pill tray works for almost everything) |
| **Different tray in each photo** | The app learned tray A but the group sits on tray B -> wrong region detected | Shoot **both** photos on the **same surface** |
| **Reference photo not a clean single pill** | App may lock onto a glare blob or shadow instead of the pill -> bad size/colour profile | Centre **one** pill, fill a good part of the frame, even lighting, no second pill in view |
| **Steep camera angle** | Pills become ellipses of varying size and cast long shadows -> size estimate drifts | Shoot **top-down** |
| **Background clutter at the photo edges** | Same-coloured objects (a hand, a white counter) near the frame edge can be mistaken for pills | Photograph against a clean tray that fills the frame, or crop the photo to the tray |
| **Very low contrast + cropped pill** | A faint pill half-cut by the frame may not be detected at all | Keep pills fully in frame and use a higher-contrast tray |
| **Mixing pill types** | Only one pill type is modelled per count -> the others may be miscounted | Count **one medication at a time** |

> **Design bias:** when the app is unsure, it is tuned to **avoid over-counting**
> (reporting more pills than exist) rather than under-counting. For medication use
> this is the safer direction, but you should **always verify against the
> annotated image**, especially for high-stakes counts.

---

## The counting algorithm

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

## How the app avoids miscounting

Counting medication is high-stakes, so the pipeline has specific safeguards
against the two ways it could go wrong. The guiding principle is: **when in
doubt, don't over-count** — for medication, reporting a phantom pill is worse
than flagging an uncertain one.

- **Scale is measured, not assumed.** The size of one pill is re-derived from the
 group photo's own isolated pills, so a reference taken at a different distance
 doesn't proportionally inflate or deflate every cluster's count.
- **Off-tray look-alikes are excluded.** Only objects sitting *inside* the
 detected tray are counted, so a same-coloured hand, counter or background near
 the frame doesn't add phantom pills.
- **Edge pills are recovered only when it's safe.** Pills touching the frame edge
 are restored **only when the tray fills the frame** (so there is no off-tray
 background to mistake for a pill). If real background reaches the edges, edge
 recovery is switched off rather than risk an over-count.
- **Clusters are counted by area, not by inflating peaks.** Touching pills are
 estimated as *total area ÷ one-pill area*, which is stable for a flat single
 layer and doesn't over-segment long capsules into extra pills.
- **Regression-tested on real photos.** The test suite includes real phone photos
 with an explicit *upper-bound* assertion (a "never over-count" guard), because
 the synthetic generator alone can't catch background-induced over-counts.

**Residual you should know about:** an extremely low-contrast pill that is also
cut off by the frame edge (e.g. a white pill, half out of frame, on a grey tray)
can occasionally be missed. This shows up as a small **under**-count, never an
over-count — another reason to keep every pill fully inside the frame.

---

## API reference

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

## Testing & accuracy

```bash
# Unit tests + real-image tests (fast, deterministic) — 27 tests
pytest test_unit.py tests/

# Self-test on the four bundled phone photos (white pills on blue tray,
# orange pills on grey tray) — prints counts vs. expected
python test_pill.py

# Synthetic accuracy harness: counts thousands of generated pills across
# many colours × shapes × counts (exact known ground truth) and reports
# exact / within-1 / within-2 rates and mean absolute error
python tests/run_accuracy.py
```

**Current accuracy** (measured, deterministic): on the synthetic matrix
(16 pill/tray colour pairs × 3 shapes × 5 counts, 240 cases): **89% exact,
96% within ±1, mean absolute error 0.44 pills.** On a set of 55 real phone
photos of labelled pill spreads (19 medication types, 8–75 pills each,
reference crops taken from the same photos): **45% within ±1 and no
catastrophic miscounts** — the hardest photos (severe defocus/motion blur,
pills with two differently-coloured faces) err by a bounded amount instead of
the order-of-magnitude failures an earlier pipeline produced. The four bundled
real photos are counted accurately (single-pill self-tests = 1, orange group
exact, white group within tolerance).

The `pytest` suite (27 tests) also covers the trickier cases directly:
- **Per-colour/shape accuracy** — white/orange/red/yellow/blue/green pills on
 contrasting trays, all within ±1.
- **Border-cropped pills** — pills touching the frame edge are recovered when the
 tray fills the frame (regression guard against the old "edge pills vanish" bug).
- **Real-photo "never over-count" guard** — the bundled phone photos are asserted
 to **not exceed** their true count, catching background-induced over-counts that
 synthetic images can't reproduce.

Beyond the committed tests, `tests/stress_probe.py` is a developer harness that
stress-tests harder conditions (cast shadows, edge-cropped pills, low contrast,
dense low-count clusters) to surface where accuracy degrades.

Test assets:
- `IMG_1241`–`IMG_1244` — real phone photos used by `test_pill.py` and the
 real-photo regression guards.
- `tests/online_images/` — freely‑licensed photos from Wikimedia Commons
 (see `tests/online_images/SOURCES.md`).
- `tests/synth.py` — the synthetic pill‑image generator used by the harness.
- `tests/stress_probe.py` — developer stress harness (not run by `pytest`).

---

## Project structure

```
ISpyPill/
├── app.py                      # Flask server: /analyze, /ping, serves the React app
├── pill_counter.py             # Core computer-vision pipeline (the algorithm)
├── requirements.txt            # Python dependencies
│
├── frontend/                   # React + Vite web app (the active UI)
│   ├── package.json            #   scripts: dev / build / preview
│   ├── vite.config.js          #   dev server :5173, proxies /analyze -> :5000
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
    ├── run_accuracy.py         # accuracy harness (240-case matrix)
    ├── stress_probe.py         # developer stress harness (shadows, edge crops, low contrast)
    ├── test_online.py          # real-image tests
    └── online_images/          # downloaded real test photos + SOURCES.md
```

> **Note:** `static/` and `templates/` contain an older vanilla‑JavaScript UI
> that has been **superseded by the React app in `frontend/`** (Flask serves
> `frontend/dist`, not these). They are kept for reference only.

---

## Troubleshooting

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
Phone and computer must be on the same Wi‑Fi, the Flask server must be running
(`python app.py`, bound to `0.0.0.0:5000`), and your firewall must allow inbound
connections to port 5000. Quick test: open `http://<your-computer-ip>:5000/ping`
in the phone's browser — it should return `{"status": "ok"}`.

**Expo Go won't open the project / "project is incompatible"**
Update Expo Go on your phone (this app targets SDK 54), then re‑scan the QR code.
On the computer, `cd mobile && npx expo-doctor` should report 18/18 checks passed;
if not, run `npm install` followed by `npx expo install --fix`.

**Tunnel mode loads the app but it says "Server unreachable"**
The Expo tunnel forwards Metro only, not the Flask backend on port 5000, so in
tunnel mode the app cannot auto‑detect the backend. Start Expo with
`EXPO_PUBLIC_API_URL` set to a URL the phone can reach (your LAN IP if on the same
Wi‑Fi, or a public URL such as a separate `ngrok http 5000` otherwise) — see
**Step 3 → Option B — tunnel** above. If the dev‑server logs print a
`[ISpyPill] Running over a tunnel … but EXPO_PUBLIC_API_URL is not set` warning,
that's exactly this case.

**Counts look off**
See [Tips for accurate counts](#tips-for-accurate-counts) — most issues are
lighting, low contrast, or stacked/overlapping pills.

---

## Limitations

- **Single layer only.** The app counts what it can see; pills stacked in a 3‑D
 pile can't all be seen, so dense heaps are under/over‑estimated.
- **Needs contrast.** Pills on a same‑coloured, textured surface with no tray
 (e.g. white capsules on white paper) are hard to separate — use a contrasting
 tray.
- **One pill type per count.** The reference defines the pill being counted;
 mixing several different pills in the group photo isn't supported.

---

## FAQ

**Q: Why do I need two photos instead of one?**
The reference (single-pill) photo is how the app works for *any* pill without
being trained on specific medications. It measures your exact pill and tray, then
finds that combination in the group photo. No reference means the app would have
to guess what a pill looks like — which is exactly what makes generic pill
counters unreliable.

**Q: Do the two photos have to be on the same tray/surface?**
Yes. The reference teaches the app what the tray looks like so it can separate
pills from it. If the group photo uses a different surface, detection will be
unreliable. Same tray, same lighting, both shot top-down.

**Q: Does it work for white pills?**
Yes — that's a core design goal. Put white pills on a **coloured or dark tray**
(a blue pill tray is ideal). The app detects pills as objects on the tray, so the
pill's own colour doesn't matter; it just needs to differ from the tray.

**Q: Can I count two different medications in one photo?**
No. Each count models one pill type (the one in your reference). Photograph and
count each medication separately.

**Q: What's the maximum number of pills I can count?**
There's no hard limit, but accuracy is best for pills spread in a **single
layer**. Dozens of well-separated pills count reliably; once pills pile up and
overlap in 3-D, hidden pills can't be seen and the count becomes an estimate.

**Q: It counted a slightly wrong number — what do I do?**
Look at the **annotated image** first: it shows exactly what was counted. Most
errors come from stacking, low contrast, or a non-contrasting surface. Re-shoot
with the pills spread out in a single layer on a contrasting tray, top-down, with
even lighting. See [Common pitfalls](#common-pitfalls).

**Q: Is it more likely to over-count or under-count?**
By design it leans toward **not over-counting** — for medication, a phantom pill
is the more dangerous error. The most likely failure is a small under-count when
pills overlap or a low-contrast pill is cut off by the frame edge. Either way,
verify against the annotated image.

**Q: Are my photos uploaded anywhere?**
No. All processing happens **locally** on the machine running the backend. There
are no accounts and nothing is sent to a cloud service. (If you deploy the
backend on a remote server, photos are sent to *that* server you control — not to
any third party.)

**Q: Should I rely on this for dispensing real medication?**
Treat it as an **assistant, not an authority.** It's a fast double-check, not a
certified counting scale. Always confirm high-stakes counts against the annotated
image and your own verification.

**Q: Do I need the mobile app, or the React web app, or both?**
Neither is required to *count* — the algorithm lives in the Python backend. The
web app (in `frontend/`) is the primary UI; the Expo app (in `mobile/`) is an
optional native client. Pick whichever fits; both call the same `/analyze`
endpoint.

**Q: Can I use the counter from my own script, without the UI?**
Yes. Import the two functions from `pill_counter.py` (see
[The counting algorithm](#the-counting-algorithm)), or POST two images to the
`/analyze` endpoint (see [API reference](#api-reference)).

**Q: Why is the count occasionally off by one on the white-on-grey test photo?**
White pills on a grey tray are the lowest-contrast case, and a pill partly cut
off by the frame can be hard to detect. This is a known low-contrast limitation
(an under-count, never an over-count). Use a more contrasting tray and keep all
pills fully in frame for best results.

**Q: The app says "React build not found" / a port is busy / the phone can't
connect.**
See [Troubleshooting](#troubleshooting) — those three are the most common setup
issues and each has a one-line fix.

---

## License

Licensed under the **MIT License** — use, modify and distribute freely,
including commercially, provided the original license is included.

---

**Happy counting!**
