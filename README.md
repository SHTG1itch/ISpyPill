# ISpyPill — Intelligent Pill Counter

A fast, accurate, and easy-to-use web application that counts pills by analyzing photos. Simply photograph a single reference pill, then photograph multiple pills together, and the application automatically counts them. Perfect for medication management, inventory, and accessibility.

**Live Demo**: Visit http://localhost:5173 or http://localhost:5000 after starting the servers (see [Quick Start](#quick-start) below).

---

## 🎯 Features

- **Single Reference Photo** — Upload one pill to establish size, color, and shape
- **Multi-Color Pill Support** — Handles single-color pills, two-tone capsules, multi-colored tablets, and more
- **Shape Validation** — Distinguishes round pills from ovals, capsules, and irregular shapes using circularity, aspect ratio, and solidity metrics
- **Automatic Separation** — Uses watershed algorithm to accurately separate touching or overlapping pills
- **Annotated Results** — View a detection map showing which regions were counted as pills
- **Mobile-Friendly** — Works on iOS and Android with responsive design, auto-rotates photos, and safe-area support
- **Fast Processing** — Optimized image resizing and efficient OpenCV operations for sub-2-second analysis
- **Accessible UI** — WCAG-compliant with keyboard navigation, proper ARIA labels, and touch-friendly targets

---

## 🔬 How It Works

### The Algorithm

ISpyPill uses a multi-stage computer vision pipeline to accurately count pills:

#### 1. **Reference Analysis** (`analyze_reference`)
When you upload a reference pill, the app:
- **Isolates the pill** using Otsu's thresholding (automatic brightness adjustment)
- **Filters contours** by area (1%-85% of image) and solidity (convexity) to find the actual pill vs. background
- **Extracts color profiles** using K-means clustering:
  - Runs K-means with K=2 and K=3 to find dominant colors
  - Auto-selects K based on compactness (variance reduction)
  - Creates a "color profile" for each cluster with:
    - HSV mean and tolerance ranges
    - Achromatic flag (for white/gray pills)
    - Hue wrap-around detection (for red pills that wrap from 0°→180°)
- **Measures shape** via:
  - **Circularity** = (4π × area) / perimeter² (1.0 = perfect circle)
  - **Aspect Ratio** = minor axis / major axis (1.0 = round, 0.5 = oval)
  - **Solidity** = area / convex hull area (1.0 = no holes)

#### 2. **Pill Counting** (`count_pills`)
When you upload a group photo:
- **Creates a color mask** by OR-ing together all reference color profiles
  - For achromatic (white) pills: uses only brightness channel (V in HSV)
  - For colored pills: uses full HSV color matching with hue wrapping
- **Applies morphological operations** (erosion → dilation) to clean noise
- **Uses watershed algorithm** to separate touching pills by distance transform
- **Filters regions** by:
  - Size (reject tiny noise that's <30% of reference area)
  - Shape (reject wrong-shaped small regions, but keep large ones even if slightly wrong shape)
  - Area ratio (if region is N× larger than reference, estimate it as ~N pills)

#### 3. **Result Visualization**
- Draws colored circles around detected pills
- Displays count with animated counter (0 → final count)
- Shows info badges: color clusters, detection mode, reference area
- Allows download of annotated image

---

## 🚀 Quick Start

### Prerequisites

You need:
- **Python 3.8+** (download from [python.org](https://www.python.org/downloads/))
- **Node.js 16+** (download from [nodejs.org](https://nodejs.org/))
- **Git** (optional, for cloning; download from [git-scm.com](https://git-scm.com/))

**Check if you have them installed:**
```bash
python --version      # Should show Python 3.8 or higher
node --version        # Should show Node.js 16 or higher
npm --version         # Should show npm 7 or higher
```

### Step 1: Get the Code

**Option A: Clone from GitHub** (if you have Git)
```bash
git clone https://github.com/yourusername/ISpyPill.git
cd ISpyPill
```

**Option B: Download ZIP** (if you don't have Git)
1. Go to https://github.com/yourusername/ISpyPill
2. Click the green "Code" button → "Download ZIP"
3. Extract the ZIP file
4. Open Terminal/Command Prompt and navigate to the extracted folder:
   ```bash
   cd path/to/ISpyPill
   ```

### Step 2: Set Up Python Backend

```bash
# Install Python dependencies
pip install flask opencv-python-headless pillow numpy

# Verify installation
python -c "import flask, cv2, numpy; print('✓ All packages installed')"
```

**On macOS/Linux**, you might need `pip3` instead:
```bash
pip3 install flask opencv-python-headless pillow numpy
```

### Step 3: Set Up React Frontend

```bash
cd frontend
npm install
cd ..
```

This downloads ~200MB of JavaScript packages — it may take 2–5 minutes.

### Step 4: Build React for Production (Optional)

If you want to use `http://localhost:5000` (Flask serving the built React app):
```bash
cd frontend
npm run build
cd ..
```

This creates `frontend/dist/` with optimized, minified code. Takes ~10–30 seconds.

### Step 5: Start the Servers

**Option A: Development Mode** (with live code reloading)

Open **two terminal windows** (one for Flask, one for React):

**Terminal 1 — Flask Backend:**
```bash
python app.py
```
You should see:
```
 * Running on http://127.0.0.1:5000
```

**Terminal 2 — React Dev Server:**
```bash
cd frontend
npm run dev
```
You should see:
```
  ➜  Local:   http://localhost:5173/
```

**Then open http://localhost:5173 in your browser.**

---

**Option B: Production Mode** (single terminal)

```bash
python app.py
```

Then open http://localhost:5000 in your browser.

> **Note**: Production mode serves the pre-built React app and doesn't auto-reload on code changes. Development mode is better for testing changes.

---

## 📱 Usage Guide

### Basic Workflow

1. **Upload a Reference Pill**
   - Click the "Gallery" or "Camera" button on the left card
   - Select or photograph a single pill that clearly fills the frame
   - The pill should be centered and well-lit for best results
   - A preview will appear once uploaded

2. **Upload a Group Photo**
   - Click the "Gallery" or "Camera" button on the right card
   - Select or photograph multiple pills spread out on a surface
   - Ensure pills are in focus and evenly lit
   - A preview will appear once uploaded

3. **Count**
   - Click the blue "Count Pills" button
   - Wait 1–3 seconds for analysis
   - View the result: pill count, detection map, and info badges

4. **Save or Retry**
   - Click "Save Image" to download the annotated image (optional)
   - Click "Count Another" to reset and start a new count

### Tips for Best Results

- **Lighting**: Use natural light or bright indoor lighting. Avoid harsh shadows.
- **Focus**: Make sure pills are in focus (not blurry).
- **Reference Pill**: Choose a pill that's fully visible and roughly centered.
- **Group Photo**: Spread pills out so they're not too densely packed.
- **Background**: Use a contrasting surface (white paper for dark pills, dark cloth for white pills).
- **Angle**: Shoot from directly above or slightly angled—avoid extreme side angles.

### Supported File Formats

- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- HEIC (`.heic`) — iPhone photos (auto-converted)

Maximum file size: 16 MB (rare for photos).

---

## 🗂️ Project Structure

```
ISpyPill/
├── README.md                      # This file
├── app.py                         # Flask backend server
├── pill_counter.py                # Core pill detection algorithm
│
├── frontend/                      # React web app
│   ├── package.json              # Node.js dependencies
│   ├── vite.config.js            # Vite build configuration
│   ├── src/
│   │   ├── main.jsx              # React entry point
│   │   ├── App.jsx               # Main app component (logic)
│   │   ├── App.css               # Global styles
│   │   └── components/
│   │       ├── UploadCard.jsx    # File upload component
│   │       ├── ResultsCard.jsx   # Results display component
│   │       └── ErrorBanner.jsx   # Error message component
│   ├── dist/                     # Built React app (created by `npm run build`)
│   └── node_modules/             # Downloaded packages (created by `npm install`)
│
└── .git/                         # Git version control (if cloned from GitHub)
```

### Key Files Explained

**`app.py`** — Flask web server
- Receives photo uploads via `/analyze` endpoint
- Calls pill detection functions
- Returns JSON with pill count and annotated image

**`pill_counter.py`** — Computer vision pipeline
- `analyze_reference()` — Extracts color, size, and shape from reference pill
- `count_pills()` — Counts pills in group photo
- Helper functions for color masking, watershed segmentation, shape validation

**`frontend/src/App.jsx`** — Main React component
- Manages upload state for both reference and group photos
- Calls `/analyze` API
- Displays results or error messages

**`frontend/src/App.css`** — Styles
- Mobile-first responsive design
- CSS variables for colors, spacing, shadows
- Safe-area support for iPhone notch

---

## 🛠️ Technical Details

### Dependencies

**Backend** (Python):
| Package | Purpose |
|---------|---------|
| `flask` | Web server framework |
| `opencv-python-headless` | Computer vision (pill detection) |
| `pillow` | Image loading and EXIF rotation |
| `numpy` | Numerical arrays and math |

**Frontend** (JavaScript):
| Package | Purpose |
|---------|---------|
| `react` | UI framework |
| `vite` | Build tool and dev server |

### Image Processing Pipeline

1. **Load & Normalize**
   - Read uploaded file as bytes
   - Detect and apply EXIF rotation (fixes iPhone photo orientation)
   - Resize to max 2000px on longest side (for speed)
   - Convert to BGR color space (OpenCV standard)

2. **Reference Analysis**
   - Apply Gaussian blur (5×5 kernel)
   - Otsu's thresholding → binary mask
   - Find contours, filter by area and solidity
   - K-means clustering for color profiles
   - Compute shape metrics

3. **Group Photo Analysis**
   - Apply same blur and color masking
   - Morphological operations (erode → dilate) for noise reduction
   - Watershed segmentation for separation
   - Find contours, validate by size and shape
   - Count and annotate

4. **Annotation**
   - Draw circles around detected pills
   - Encode as base64 JPEG (85% quality)
   - Return to frontend for display

### Color Detection Strategy

**For single-color pills:**
- Extract mean and std of each HSV channel from pill pixels
- Create upper/lower bounds: `mean ± 2×std` (captures ~95% of color variation)

**For multi-color pills:**
- Run K-means clustering with K=2 and K=3
- Choose K based on compactness improvement (K=3 only if >40% reduction)
- For achromatic clusters (saturation <40), use only brightness (V channel)

**For hue wrap-around** (red pills that cross 0°/180° boundary):
- Detect when cluster center is <10° or >165°
- Split matching into two `cv2.inRange` calls
- OR the results together

### Shape Validation

Detected regions are validated by comparing shape metrics to reference:

| Metric | Tolerance | Purpose |
|--------|-----------|---------|
| Circularity | ±0.35 (scaled) | Rejects thin noise |
| Aspect Ratio | ±0.30 (scaled) | Rejects elongated artifacts |
| Solidity | ±0.25 (scaled) | Rejects concave shapes |

Tolerance scales with cluster size: `tolerance * sqrt(N)` where N is estimated pill count.

---

## 🐛 Troubleshooting

### "React build not found" Error

**Problem:** When running `python app.py`, you see:
```
⚠  React build not found.
   Run:  cd frontend && npm install && npm run build
```

**Solution:**
```bash
cd frontend
npm install
npm run build
cd ..
python app.py
```

---

### "ModuleNotFoundError: No module named 'flask'"

**Problem:** Python can't find the Flask package.

**Solution:**
```bash
pip install flask opencv-python-headless pillow numpy
```

On macOS/Linux, use `pip3`:
```bash
pip3 install flask opencv-python-headless pillow numpy
```

---

### "Port 5000 already in use"

**Problem:** Another app is using port 5000.

**Solution — Option A:** Find and stop the other app:
```bash
# On Windows (in PowerShell):
netstat -ano | findstr :5000

# On macOS/Linux:
lsof -i :5000
```

Then kill the process or use a different port:
```bash
python app.py --port 5001
```

**Solution — Option B:** Restart your computer (resets all ports).

---

### Pills Are Miscounted or Not Detected

**Common Causes & Fixes:**

1. **Poor lighting** → Retake photo in brighter light, avoid shadows
2. **Blurry photo** → Ensure camera is in focus before taking photo
3. **Reference pill not centered** → Retake reference with pill more centered
4. **Pills too close together** → Spread them out in group photo
5. **Wrong surface** → Try a contrasting background (white for dark pills, dark for white pills)

**Debugging:** Click "Save Image" to see the detection map. Green circles show detected pills. If circles are missing or wrong:
- The color might not match reference (e.g., reference was shiny, group photo is matte)
- The shape might be too different
- Try a different reference pill or adjust lighting

---

### "TypeError: unsupported operand type" or "OpenCV assertion failed"

**Problem:** You see an error like:
```
cv2.error: OpenCV(4.5.0) ... cv2.inRange() ... assertion failed
```

**Cause:** Usually a dtype (data type) mismatch in image arrays.

**Solution:** Reinstall OpenCV:
```bash
pip install --upgrade opencv-python-headless
```

---

### App Runs But Returns "Image processing failed"

**Problem:** Upload succeeds, but server returns a 500 error.

**Cause:** An edge case in pill detection (e.g., no pills detected, invalid image).

**Solution:**
1. Check the terminal running Flask for error details
2. Try a clearer, better-lit photo
3. Make sure reference pill is fully visible and in focus

---

## 💻 Development

### Building & Testing

**Run tests** (if included):
```bash
pytest tests/
```

**Build for production:**
```bash
cd frontend
npm run build    # Creates optimized dist/ folder
cd ..
python app.py    # Serves from dist/
```

**Development with live reload:**
- Terminal 1: `python app.py`
- Terminal 2: `cd frontend && npm run dev`
- Edit code → changes appear immediately at http://localhost:5173

### Adding Features

The codebase is organized for easy modification:

- **Change UI**: Edit `frontend/src/App.jsx` and `frontend/src/App.css`
- **Add new components**: Create `.jsx` files in `frontend/src/components/`
- **Modify detection algorithm**: Edit `pill_counter.py`
- **Change API responses**: Edit the `/analyze` route in `app.py`

### Code Quality

- **Python**: Follow PEP 8 conventions. Use descriptive variable names.
- **JavaScript/React**: Use modern ES6+ syntax. Prefer functional components with hooks.
- **CSS**: Use CSS variables for theming. Mobile-first approach.

---

## 📦 Deployment

### Deploy to the Cloud

**Heroku** (free tier available):
1. Create account at heroku.com
2. Install Heroku CLI
3. Run:
   ```bash
   heroku login
   heroku create my-pill-counter
   git push heroku main
   ```

**AWS, Google Cloud, or DigitalOcean**:
- Docker: Create a `Dockerfile` that installs Python + Node dependencies and runs both servers
- Recommend using Gunicorn for Flask instead of dev server

### Environment Variables

If deploying, consider setting:
- `FLASK_ENV=production` (disables debug mode)
- `PORT=5000` (for cloud platforms that assign a port)

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. **Fork** the repository on GitHub
2. **Create a branch**: `git checkout -b feature/my-feature`
3. **Make changes** and test locally
4. **Commit**: `git commit -m "Add my feature"`
5. **Push**: `git push origin feature/my-feature`
6. **Open a Pull Request** with a description of changes

### Reporting Issues

If you find a bug:
1. Describe what happened and how to reproduce it
2. Include the photo if possible (sanitized, no personal data)
3. Share the error message from the terminal
4. Open an issue on GitHub

---

## 📝 License

This project is licensed under the **MIT License**. See `LICENSE` file for details.

In short: You can use, modify, and distribute this project freely, including for commercial purposes, as long as you include the original license.

---

## 🎓 Learning Resources

### Computer Vision Concepts

If you want to understand the algorithm better:

- **OpenCV Tutorials**: https://docs.opencv.org/master/
- **K-means Clustering**: https://en.wikipedia.org/wiki/K-means_clustering
- **Watershed Algorithm**: https://docs.opencv.org/master/d3/db0/tutorial_js_watershed.html
- **HSV Color Space**: https://en.wikipedia.org/wiki/HSL_and_HSV

### Web Development

- **Flask Docs**: https://flask.palletsprojects.com/
- **React Docs**: https://react.dev/
- **Vite Docs**: https://vitejs.dev/

---

## ❓ FAQ

**Q: Can I count pills that aren't pills?**
A: Yes! Any round/oval object similar in size to the reference will be counted. Works with coins, candies, etc.

**Q: What's the accuracy?**
A: Typically 95%+ for well-lit, evenly spaced pills. Accuracy drops if pills are touching, partially obscured, or under poor lighting.

**Q: How long does analysis take?**
A: 0.5–2 seconds depending on image size and number of pills.

**Q: Can I use this on my phone?**
A: Yes! Open http://yourcomputerip:5173 on your phone (replace `yourcomputerip` with your computer's IP). Find IP:
  - **Windows**: Run `ipconfig`, look for "IPv4 Address"
  - **macOS/Linux**: Run `ifconfig`, look for "inet"

**Q: Is my photo data sent to the cloud?**
A: No. Everything runs locally on your computer. Photos are never uploaded anywhere.

**Q: Can I deploy this online?**
A: Yes, see [Deployment](#deployment) section.

---

## 📞 Support

- **Issues**: Open an issue on GitHub
- **Questions**: Start a Discussion on GitHub
- **Email**: [Add contact email if desired]

---

## 🙏 Acknowledgments

- Built with [Flask](https://flask.palletsprojects.com/) and [React](https://react.dev/)
- Computer vision powered by [OpenCV](https://opencv.org/)
- Inspired by medication management accessibility needs

---

## 📊 Project Stats

- **Lines of Code**: ~800 (Python) + ~1200 (JavaScript)
- **Key Algorithms**: K-means clustering, Watershed segmentation, Contour analysis
- **Supported Formats**: JPEG, PNG, HEIC
- **Max Image Size**: 16 MB
- **Processing Time**: <2 seconds per image
- **Browser Support**: All modern browsers (Chrome, Safari, Firefox, Edge)

---

**Happy counting! 💊**

If you find this project useful, consider giving it a star on GitHub ⭐
