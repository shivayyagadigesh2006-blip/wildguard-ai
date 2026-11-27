"""Memory-conscious whole-image species classification API for Render."""

import asyncio
import json
import math
import os
import sqlite3
import threading
from contextlib import asynccontextmanager, closing
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, BinaryIO

import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import nn
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_wild_animal_classifier_b3.pth"
DATABASE_PATH = Path(
    os.getenv("DATABASE_PATH", str(ROOT / "data" / "wildguard.sqlite3"))
).resolve()
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 6_000_000
MIN_CONFIDENCE = 0.5
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

torch.set_num_threads(1)

_MODEL: nn.Module | None = None
_CLASS_NAMES: tuple[str, ...] = ()
_TRANSFORM: transforms.Compose | None = None
_INFERENCE_LOCK = threading.Lock()


def _load_classifier() -> tuple[nn.Module, tuple[str, ...], transforms.Compose]:
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Wildlife classifier weights are missing: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("The classifier checkpoint has an unsupported format.")

    state_dict = checkpoint.get("model_state_dict")
    class_names = checkpoint.get("class_names")
    resolution = checkpoint.get("input_resolution")
    if not isinstance(state_dict, dict) or not isinstance(class_names, list) or not class_names:
        raise RuntimeError("The classifier checkpoint is missing its weights or class names.")
    if (
        not isinstance(resolution, (list, tuple))
        or len(resolution) != 2
        or not all(isinstance(value, int) and value > 0 for value in resolution)
    ):
        raise RuntimeError("The classifier checkpoint has an invalid input resolution.")

    class_names = tuple(str(name) for name in class_names)
    model = models.efficientnet_b3(weights=None)
    hidden_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(hidden_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, len(class_names)),
    )
    model.half()
    model.load_state_dict(state_dict, strict=True)
    del checkpoint, state_dict
    model.eval()

    image_transform = transforms.Compose(
        [
            transforms.Resize(tuple(resolution)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return model, class_names, image_transform


def _initialize_database() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DATABASE_PATH)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    predictions_json TEXT NOT NULL,
                    candidates_json TEXT NOT NULL,
                    sampled_frames INTEGER NOT NULL,
                    detections_json TEXT NOT NULL DEFAULT '[]',
                    latitude REAL,
                    longitude REAL,
                    location_source TEXT
                )
                """
            )
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(analyses)").fetchall()
        }
        migrations = {
            "detections_json": "TEXT NOT NULL DEFAULT '[]'",
            "latitude": "REAL",
            "longitude": "REAL",
            "location_source": "TEXT",
        }
        for column, definition in migrations.items():
            if column not in columns:
                connection.execute(f"ALTER TABLE analyses ADD COLUMN {column} {definition}")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    global _MODEL, _CLASS_NAMES, _TRANSFORM
    _initialize_database()
    _MODEL, _CLASS_NAMES, _TRANSFORM = _load_classifier()
    yield
    _MODEL = None
    _CLASS_NAMES = ()
    _TRANSFORM = None


app = FastAPI(title="WildGuard AI Species Classifier", version="1.0.0", lifespan=lifespan)

cors_origins = {"https://wildguard-ai-frontend.vercel.app"}
cors_origins.update(
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://127.0.0.1:8001,http://localhost:8001",
    ).split(",")
    if origin.strip()
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _connect_database() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def _display_name(class_name: str) -> str:
    return class_name.replace("_", " ").title()


def _predict(image: Image.Image) -> list[dict[str, object]]:
    if _MODEL is None or _TRANSFORM is None:
        raise RuntimeError("The wildlife classifier is not ready.")
    tensor = _TRANSFORM(image).unsqueeze(0).to(dtype=torch.float16)
    with _INFERENCE_LOCK, torch.inference_mode():
        scores = torch.softmax(_MODEL(tensor), dim=1)[0]
    top_indices = torch.topk(scores, k=min(5, len(_CLASS_NAMES))).indices.tolist()
    return [
        {
            "species": _display_name(_CLASS_NAMES[index]),
            "scientific": "",
            "confidence": float(scores[index].item()),
            "count": 1,
        }
        for index in top_indices
    ]


def _load_image(file: BinaryIO) -> Image.Image:
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The image exceeds the 15 MB upload limit.")

    try:
        with Image.open(file) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="The image dimensions exceed the supported 6-megapixel limit.",
                )
            return ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=422, detail="The uploaded image could not be decoded.") from exc


def _record_analysis(name: str, result: dict[str, object]) -> None:
    location = result["location"]
    if not isinstance(location, dict):
        location = {}
    with closing(_connect_database()) as connection:
        with connection:
            connection.execute(
                """
                INSERT INTO analyses
                    (name, media_type, created_at, predictions_json, candidates_json, sampled_frames,
                     detections_json, latitude, longitude, location_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name[:255],
                    "image",
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    json.dumps(result["predictions"]),
                    json.dumps(result["candidates"]),
                    result["sampled_frames"],
                    "[]",
                    location.get("latitude"),
                    location.get("longitude"),
                    location.get("source"),
                ),
            )


@app.get("/api/health")
def health() -> dict[str, object]:
    classifier_ready = _MODEL is not None
    return {
        "status": "ok" if classifier_ready else "not_ready",
        "capabilities": {
            "detect": False,
            "species": classifier_ready,
            "audio": False,
            "assistant": False,
        },
        "message": (
            "Whole-image species classifier is configured; object detection is not included."
            if classifier_ready
            else "The wildlife classifier is not loaded."
        ),
        "classes": len(_CLASS_NAMES),
        "minimum_confidence": MIN_CONFIDENCE,
    }


@app.get("/api/dashboard")
def dashboard() -> dict[str, object]:
    with closing(_connect_database()) as connection:
        rows = connection.execute(
            "SELECT created_at, predictions_json FROM analyses ORDER BY created_at"
        ).fetchall()

    total_predictions = 0
    species: set[str] = set()
    confidences: list[float] = []
    daily: dict[str, int] = {}
    for row in rows:
        predictions = json.loads(row["predictions_json"])
        day = row["created_at"][:10]
        total_predictions += len(predictions)
        species.update(str(prediction["species"]) for prediction in predictions)
        confidences.extend(float(prediction["confidence"]) for prediction in predictions)
        daily[day] = daily.get(day, 0) + len(predictions)

    return {
        "total_predictions": total_predictions,
        "species_identified": len(species),
        "analysis_sessions": len(rows),
        "average_confidence": sum(confidences) / len(confidences) if confidences else None,
        "daily": [{"date": date, "count": count} for date, count in sorted(daily.items())],
    }


@app.get("/api/history")
def history(limit: int = Query(default=6, ge=1, le=100)) -> list[dict[str, object]]:
    with closing(_connect_database()) as connection:
        rows = connection.execute(
            "SELECT name, media_type, predictions_json FROM analyses ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

    entries = []
    for row in rows:
        predictions = json.loads(row["predictions_json"])
        top = predictions[0] if predictions else {}
        entries.append(
            {
                "name": row["name"],
                "type": row["media_type"].title(),
                "top": top.get("species", "Uncertain"),
                "sci": "",
                "conf": round(float(top.get("confidence", 0)) * 100),
                "status": "review",
            }
        )
    return entries


@app.get("/api/locations")
def locations(limit: int = Query(default=1000, ge=1, le=5000)) -> list[dict[str, object]]:
    with closing(_connect_database()) as connection:
        rows = connection.execute(
            "SELECT id, name, created_at, latitude, longitude, location_source, predictions_json "
            "FROM analyses WHERE latitude IS NOT NULL AND longitude IS NOT NULL "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {
            "id": int(row["id"]),
            "name": row["name"],
            "created_at": row["created_at"],
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "source": row["location_source"],
            "predictions": json.loads(row["predictions_json"]),
        }
        for row in rows
    ]


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    browser_latitude: float | None = Form(default=None),
    browser_longitude: float | None = Form(default=None),
) -> dict[str, object]:
    filename = Path(file.filename or "upload").name
    if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
        await file.close()
        raise HTTPException(status_code=415, detail="Upload a supported image file.")
    if (browser_latitude is None) != (browser_longitude is None):
        await file.close()
        raise HTTPException(status_code=422, detail="Both browser location coordinates are required.")
    if browser_latitude is not None and (
        not math.isfinite(browser_latitude)
        or not math.isfinite(browser_longitude)
        or abs(browser_latitude) > 90
        or abs(browser_longitude) > 180
    ):
        await file.close()
        raise HTTPException(status_code=422, detail="The browser location coordinates are invalid.")

    try:
        image = _load_image(file.file)
        candidates = await asyncio.to_thread(_predict, image)
    finally:
        await file.close()

    location = (
        {
            "latitude": browser_latitude,
            "longitude": browser_longitude,
            "source": "browser upload location",
        }
        if browser_latitude is not None and browser_longitude is not None
        else None
    )
    result: dict[str, object] = {
        "predictions": candidates[:1],
        "candidates": candidates,
        "detections": [],
        "sampled_frames": 1,
        "image_size": [image.width, image.height],
        "location": location,
        "media_type": "image",
        "note": (
            "Whole-image species classification only; this model does not locate individual animals "
            "or produce bounding boxes. Predictions below 50% confidence need review."
        ),
    }
    _record_analysis(filename, result)
    return result
