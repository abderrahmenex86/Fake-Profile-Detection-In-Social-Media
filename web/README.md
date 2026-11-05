# Browser demo

Live at <https://fpd.abderrahmene.me>, deployed on Vercel.

A static site that runs the benchmark's best model entirely in the browser: HOG features plus the 200-tree XGBoost ensemble, no server, no upload.

```text
web/
├── index.html
├── styles.css
├── app.js          image decoding, UI, drag/drop/paste
├── hog.js          resize, grayscale, HOG, block normalization
├── model.js        scaler, tree traversal, softmax
├── assets/
│   ├── model.json  trees, class offsets, feature config, test metrics
│   └── scaler.bin  float32 mean / scale, 34,992 features
├── samples/        four test-split screenshots
└── fonts/          self-hosted woff2 subsets
```

## Run it

```bash
make web
# or
python3 -m http.server 8799 --directory web
```

Then open <http://localhost:8799>. A plain `file://` open will not work because ES modules and `fetch` need HTTP.

Deep link a sample with `?sample=bot`, `?sample=cyborg`, `?sample=real`, or `?sample=verified`.

## How it works

1. `createImageBitmap` decodes the file, and `getImageData` yields RGBA pixels.
2. `areaResize` reproduces OpenCV's `INTER_AREA` resize to 450×600 (float32 weights, half-even rounding), then `toGray` uses OpenCV's 8-bit coefficients.
3. `extractCells` computes gradients, unsigned orientation in degrees, and 9-bin histograms per 16 px cell, averaged over the cell.
4. `normalizeBlocks` applies the L2-Hys normalization over 2×2 cell blocks and flattens to 34,992 features.
5. `standardize` applies the exported scaler, `predict` walks the 200 trees (50 per class), adds the class offsets, and applies softmax.

The HOG plate is drawn from the same cell histograms that feed the model, so it shows what the classifier sees rather than a separate visualization.

## Rebuild the artifacts

```bash
python scripts/export_web_model.py
```

This reads `models/hog_xgb/model.ubj`, verifies that the exported trees reproduce XGBoost's margins and probabilities, and writes `assets/model.json` + `assets/scaler.bin`.

## Validation

`scripts/validate_web_model.py` writes reference features and probabilities for a random sample of the test split, and `scripts/validate_web.mjs` replays the same images through the browser code in Node:

```bash
python scripts/validate_web_model.py --count 250
node scripts/validate_web.mjs /tmp/opencode/web-validation-250
```

On 250 test images: label agreement 250/250, mean feature difference 1.4e-5, maximum probability difference 0.05. The remaining differences come from a handful of pixels where OpenCV's resize rounds half-way values differently from float32 accumulation in JS; they do not change a prediction in the sample.

## Deploy

Any static host works, because there is no backend. Keep the directory structure.

**Vercel** — the production deployment. Import the repo, set the root directory to `web`, framework preset `Other`, no build command, then add the `fpd.abderrahmene.me` domain and a `fpd` CNAME to `cname.vercel-dns.com` at the registrar.

**Cloudflare Pages** — free, no build step:

```bash
npx wrangler pages deploy web --project-name fake-profile-detection
```

Or drag the `web` folder into the Pages dashboard.

**GitHub Pages** — publish the folder with an Actions workflow (Pages source: GitHub Actions):

```yaml
name: pages
on:
  push:
    branches: [main]
permissions:
  contents: read
  pages: write
  id-token: write
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: github-pages
    steps:
      - uses: actions/checkout@v4
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: web
      - uses: actions/deploy-pages@v4
```

**Hugging Face static Space** — create a Space with SDK `static`, then push the contents of `web/` to its root, with a `README.md` whose front matter is `sdk: static`. The model repo can stay separate; link the two.

## Limits

- Input is stretched to 450×600, exactly as in training. Cropped or very different layouts are out of distribution.
- Lossy compression and resampling can flip a confident prediction. One BOT sample scores 0.999 as a 450×600 PNG, 0.898 CYBORG after a round trip through 600×800; Python does the same, so this is the model, not the port. The bundled samples are lossless PNGs at 450×600.
- The classifier knows four synthetic classes from one dataset. It is a benchmark demo, not a moderation verdict.
