# Online test images

Real pill photographs downloaded from **Wikimedia Commons** (freely licensed)
to validate the counter on real-world images, complementing the synthetic
accuracy matrix and the in-repo phone photos (`IMG_124x`).

| File | Description | Ground truth | Notes |
|------|-------------|--------------|-------|
| `Generic_12-Hour_Allergy_Pills.JPG` | 6 amber tablets, spread out on dark wood | **6** | Counted exactly. Representative of intended use (pills laid out in a single layer). |
| `Equate_Ibuprofen_Pills.JPG` | ~45 orange caplets in a dense, **stacked 3-D heap** on wood | ambiguous | Out-of-spec stress case: pills overlap and stack, so an exact count is not well-defined and the counter only degrades gracefully (it detects a large pile, not an exact number). The app is designed for pills spread in a single layer. |

`*_REF.jpg` are single-pill reference crops taken from the same photo (the app
needs one reference pill of the same type as the group).

Source: https://commons.wikimedia.org/wiki/Category:Pills_on_tables
(uploads by user *LadyofProcrastination*). These are used here only as
read-only test fixtures.

Note: the Chrome extension was not connected in the working session, so images
were fetched via Wikimedia's `Special:FilePath` endpoint instead of the browser.
