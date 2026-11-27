# WildGuard AI

## Run locally

From the repository root, create and activate a virtual environment, install the backend dependencies, and download the detector weights:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements-full.txt
python backend/download_detector.py
python backend/full_app.py
```

Open [http://127.0.0.1:8001](http://127.0.0.1:8001). Use this address rather than opening `index.html` directly or serving it from a separate Live Server; the Python server serves the page, its JavaScript and CSS, and the `/api` endpoints from one origin. The 3D Studio additionally loads three.js and its model assets from the configured CDN.

Animal detection and species classification use the local models; uploaded images and video frames are processed locally.

## Render PTH classifier

The Render Blueprint runs `backend/classifier_app.py` through `backend/app.py` and uses only the uploaded `backend/models/best_wild_animal_classifier_b3.pth` classifier. It installs CPU-only PyTorch, limits inference to one thread, and uses half precision to reduce memory. It does not need Modal or an external inference URL.

This checkpoint classifies one whole image among its 27 trained classes. It does not locate animals, return bounding boxes, or analyze video. Uploads are limited to 15 MB and 6 megapixels. The capability status therefore reports species classification as available and object detection as unavailable. The Render free service has ephemeral storage, so its SQLite history can be lost after a restart or redeploy.

Deploy with [render.yaml](./render.yaml). The Render service allows the Vercel origin `https://wildguard-ai-frontend.vercel.app` through `CORS_ORIGINS`.

To run the full AI backend locally, install the full requirements and model files:

```powershell
python -m pip install -r backend/requirements-full.txt
python backend/download_detector.py
python backend/full_app.py
```

## Deploy the frontend with Vercel

Vercel hosts the static frontend; Render runs the lightweight PTH classifier API.

1. The Render Blueprint sets `CORS_ORIGINS` to `https://wildguard-ai-frontend.vercel.app`.
2. Import this repository into Vercel. It serves `index.html`, CSS, JavaScript and public assets as static files.
3. The Render backend URL is configured in `frontend/index.html`:

```html
<meta name="wildguard-api-base" content="https://wildguard-ai-q2q3.onrender.com" />
```

Alternatively, set `window.WILDGUARD_API_BASE` before `frontend/Script.js` loads. Do not add `/api` to the URL.
## Multi-animal detection and species classification

The server uses Microsoft's MegaDetector V6 RT-DETR detector (`MDV6-apa-rtdetr-e`, Apache-2.0) to find animal bounding boxes, then classifies each detected crop with `backend/models/best_wild_animal_classifier_b3.pth`. Download the approximately 278 MB detector package with `python backend/download_detector.py` before starting the server. The helper verifies the download against the checksum published by the official SPARROW-Engine model catalog and stores it under `backend/models/sparrow/`.

Videos are sampled at up to 30 evenly spaced frames. Results show each detected animal instance, its bounding box, detector score, crop classification, and sampled-frame time; detections are not tracked or deduplicated into individual animals across frames. The classifier has 27 classes; animals outside those classes may receive an incorrect label or need review. The checkpoint has no validation accuracy or training/preprocessing code. Its 512-unit ReLU/dropout head was reconstructed from the saved tensor shapes and uses 300×300 resizing with ImageNet normalization. Neither model's confidence is a measured accuracy guarantee; verify results, especially for unfamiliar species or poor-quality media.

## Geotags and privacy

When Analyze is clicked, the browser requests the current location. If permission is granted, that upload-time location is plotted on the Leaflet map; if permission is denied or unavailable, animal analysis still runs without a map point. The map refreshes recorded upload observations every 30 seconds; it does not identify or track an individual animal. Browser coordinates and detections are stored in the local `backend/data/wildguard.sqlite3` database, and uploaded media is not retained.

Leaflet is loaded from unpkg and map tiles from the public OpenStreetMap tile service. Tile requests reveal the viewed map area and your IP address to that service. The map displays OpenStreetMap attribution and requests only normal visible tiles; the public tile service is best-effort, not intended for high-volume production use. Use a self-hosted or commercial tile provider if traffic grows. Keep this server bound to `127.0.0.1` and protect the database if the computer is shared. Map markers show upload locations, not the animal's exact position.

The supplied EfficientNet-B3 checkpoint still does not include a validation set, so species scores are model outputs rather than measured accuracy. Missing model files or incompatible checkpoints cause an explicit startup error instead of fabricated predictions.
# wildguard-ai
# wildguard-ai
# wildguard-ai
# wildguard-ai
# wildguard-ai
# wildguard-ai
# wildguard-ai
