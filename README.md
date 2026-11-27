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

## Render API

The Render Blueprint uses the lightweight `backend/app.py` and `backend/requirements.txt`; Torch, OpenCV and the model weights stay off the 512 MiB Render service. When configured, it proxies `/api/analyze` to Modal and records returned results in its SQLite database. Deploy with [render.yaml](./render.yaml). It allows the Vercel origin `https://wildguard-ai-frontend.vercel.app` through `CORS_ORIGINS`.

The Render free service has ephemeral storage, so its SQLite history can be lost after a restart or redeploy. Add these Render environment variables after deploying the Modal inference app:

- `INFERENCE_API_URL`: the Modal endpoint base URL, without `/api`
- `INFERENCE_API_KEY`: the same secret used by the Modal app

## Deploy AI inference with Modal

From the repository root, install the Modal CLI and authenticate:

```powershell
python -m pip install -r backend/requirements-modal.txt
modal setup
```

Create a private key and store it as a Modal secret:

```powershell
$secret = python -c "import secrets; print(secrets.token_urlsafe(32))"
modal secret create wildguard-inference-key "INFERENCE_API_KEY=$secret"
```

Deploy the inference service:

```powershell
modal deploy backend/modal_inference.py
```

The deploy output includes the Modal API base URL. Set that URL as Render's `INFERENCE_API_URL` and the same generated key as `INFERENCE_API_KEY`, then redeploy Render. Modal runs the full detector and classifier with 4 GiB memory; the model image is built separately from Render and Modal usage may incur charges. The Modal API requires the shared key and the Render proxy supplies it automatically.

To run the full AI backend locally, install the full requirements and model files:

```powershell
python -m pip install -r backend/requirements-full.txt
python backend/download_detector.py
python backend/full_app.py
```

## Deploy the frontend with Vercel

WildGuard's ML inference is a full Python process and should run on Render (or another long-running Python host). Vercel can host the static frontend, but a Vercel serverless function is not suitable for the current Torch/ONNX models and model size.

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
