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

## Lightweight Render deployment

The Render Blueprint uses the lightweight `backend/app.py` and `backend/requirements.txt`. It avoids downloading the 286 MB detector and does not install Torch, OpenCV, or the full AI dependency stack. The lightweight service keeps the health, dashboard, history, and locations endpoints available, but `/api/analyze` responds with HTTP 503 because inference is disabled. `app.py` is deliberately lightweight so a Render service using the common `uvicorn app:app` command will not load the AI models.

Deploy with Render Blueprint using [render.yaml](./render.yaml). Set `CORS_ORIGINS` to your Vercel origin (for example `https://wildguard-ai.vercel.app`). A Render free service has ephemeral storage, so its SQLite history can be lost when the service restarts or redeploys.

To run the full AI backend locally, install the full requirements and model files:

```powershell
python -m pip install -r backend/requirements-full.txt
python backend/download_detector.py
python backend/full_app.py
```

The full backend includes Torch and both wildlife models and therefore exceeds the lightweight Render footprint.

## Deploy the frontend with Vercel

WildGuard's ML inference is a full Python process and should run on Render (or another long-running Python host). Vercel can host the static frontend, but a Vercel serverless function is not suitable for the current Torch/ONNX models and model size.

1. Set `CORS_ORIGINS` on Render to your Vercel origin, for example `https://wildguard-ai.vercel.app`.
2. Import this repository into Vercel. It serves `index.html`, CSS, JavaScript and public assets as static files.
3. Set the Render backend URL in `frontend/index.html`:

```html
<meta name="wildguard-api-base" content="https://your-service.onrender.com" />
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
