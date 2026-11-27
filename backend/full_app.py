"""Full local FastAPI server for WildGuard AI and its wildlife classifier."""

import asyncio
import json
import logging
import math
import os
import re
import sqlite3
import threading
import tempfile
from contextlib import asynccontextmanager, closing
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Mapping

import av
import cv2
import numpy as np
import sparrow_engine
import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError
from sparrow_engine import BBox, Detection, DetectResult
from torchvision import models, transforms
from torch import nn

ROOT = Path(__file__).resolve().parent
FRONTEND_ROOT = ROOT.parent / "frontend"
MODEL_PATH = ROOT / "models" / "best_wild_animal_classifier_b3.pth"
SPARROW_MODEL_DIR = ROOT / "models" / "sparrow"
DETECTOR_MODEL_ID = "MDV6-apa-rtdetr-e"
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(ROOT / "data" / "wildguard.sqlite3"))).resolve()
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_VIDEO_FRAMES = 30
MAX_DETECTIONS_PER_FRAME = 100
CLASSIFIER_BATCH_SIZE = 32
DETECTOR_MIN_CONFIDENCE = 0.5
DETECTOR_NMS_IOU_THRESHOLD = 0.4
MIN_CONFIDENCE = 0.5

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
ISO6709_PATTERN = re.compile(
    r"^\s*([+-]\d{2}(?:\.\d+)?)([+-]\d{3}(?:\.\d+)?)(?:[+-]\d{2,}(?:\.\d+)?)?/?\s*$"
)
STATIC_FILES = {
    "index.html": FRONTEND_ROOT / "index.html",
    "Style.css": FRONTEND_ROOT / "Style.css",
    "Script.js": FRONTEND_ROOT / "Script.js",
    "Studio.js": FRONTEND_ROOT / "Studio.js",
}
STATIC_DIRECTORIES = {"images", "models"}
STATIC_EXTENSIONS = {".avif", ".glb", ".jpeg", ".jpg", ".png", ".webp"}

_MODEL: nn.Module | None = None
_CLASS_NAMES: tuple[str, ...] = ()
_TRANSFORM: transforms.Compose | None = None
_DETECTOR_READY = False
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_INFERENCE_LOCK = threading.Lock()
_LOGGER = logging.getLogger(__name__)


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
    model.load_state_dict(state_dict, strict=True)
    model.to(_DEVICE)
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


def _load_detector() -> None:
    global _DETECTOR_READY
    model_directory = SPARROW_MODEL_DIR / DETECTOR_MODEL_ID
    manifest_path = model_directory / "manifest.toml"
    if not manifest_path.is_file() or not any(model_directory.rglob("*.onnx")):
        _LOGGER.warning(
            "Detector weights are missing at %s; downloading the verified model package.",
            model_directory,
        )
        from download_detector import install_detector

        install_detector()
    sparrow_engine.init(device="auto", model_dir=str(SPARROW_MODEL_DIR))
    available_models = sparrow_engine.list_models()
    if not any(model.id == DETECTOR_MODEL_ID for model in available_models):
        raise RuntimeError(f"The detector model {DETECTOR_MODEL_ID} is not available.")
    _DETECTOR_READY = True


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    global _MODEL, _CLASS_NAMES, _TRANSFORM, _DETECTOR_READY
    _initialize_database()
    _MODEL, _CLASS_NAMES, _TRANSFORM = _load_classifier()
    _load_detector()
    yield
    _MODEL = None
    _CLASS_NAMES = ()
    _TRANSFORM = None
    _DETECTOR_READY = False


app = FastAPI(title="WildGuard AI", version="1.0.0", lifespan=lifespan)

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://127.0.0.1:8001,http://localhost:8001",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _display_name(class_name: str) -> str:
    return class_name.replace("_", " ").title()


def _predict(images: list[Image.Image]) -> torch.Tensor:
    if _MODEL is None or _TRANSFORM is None:
        raise RuntimeError("The wildlife classifier is not ready.")
    if not images:
        return torch.empty((0, len(_CLASS_NAMES)))
    scores = []
    for start in range(0, len(images), CLASSIFIER_BATCH_SIZE):
        batch = torch.stack(
            [_TRANSFORM(image) for image in images[start : start + CLASSIFIER_BATCH_SIZE]]
        ).to(_DEVICE)
        with torch.inference_mode():
            scores.append(torch.softmax(_MODEL(batch), dim=1).cpu())
    return torch.cat(scores)


def _coordinates_from_gps_ifd(gps: Mapping[int, object]) -> dict[str, float] | None:
    def coordinate(value: object, reference: object, positive: str, negative: str, limit: float) -> float | None:
        if not isinstance(value, (tuple, list)) or len(value) != 3:
            return None
        if isinstance(reference, bytes):
            reference = reference.decode("ascii", errors="ignore")
        if not isinstance(reference, str) or reference.upper() not in {positive, negative}:
            return None
        try:
            degrees, minutes, seconds = (float(part) for part in value)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return None
        result = abs(degrees) + abs(minutes) / 60 + abs(seconds) / 3600
        if not math.isfinite(result) or result > limit:
            return None
        return -result if reference.upper() == negative else result

    latitude = coordinate(gps.get(2), gps.get(1), "N", "S", 90)
    longitude = coordinate(gps.get(4), gps.get(3), "E", "W", 180)
    if latitude is None or longitude is None:
        return None
    return {"latitude": latitude, "longitude": longitude}


def _coordinates_from_video_metadata(
    metadata_groups: list[Mapping[str, object]],
) -> dict[str, float] | None:
    latitude: float | None = None
    longitude: float | None = None
    for metadata in metadata_groups:
        for key, raw_value in metadata.items():
            key_normalized = str(key).lower()
            value = str(raw_value).strip()
            if "iso6709" in key_normalized or "location" in key_normalized:
                match = ISO6709_PATTERN.match(value)
                if match:
                    coordinates = _valid_coordinates(float(match.group(1)), float(match.group(2)))
                    if coordinates is not None:
                        return coordinates
            compact_key = re.sub(r"[^a-z]", "", key_normalized)
            if "latitude" in compact_key or compact_key == "gpslat":
                try:
                    latitude = float(value)
                except ValueError:
                    continue
            elif "longitude" in compact_key or compact_key == "gpslon":
                try:
                    longitude = float(value)
                except ValueError:
                    continue
    if latitude is None or longitude is None:
        return None
    return _valid_coordinates(latitude, longitude)


def _valid_coordinates(latitude: float, longitude: float) -> dict[str, float] | None:
    if (
        not math.isfinite(latitude)
        or not math.isfinite(longitude)
        or abs(latitude) > 90
        or abs(longitude) > 180
    ):
        return None
    return {"latitude": latitude, "longitude": longitude}


def _load_image(path: Path) -> tuple[Image.Image, dict[str, float] | None]:
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            try:
                gps = exif.get_ifd(0x8825) if 0x8825 in exif else {}
            except (OSError, TypeError, ValueError) as exc:
                _LOGGER.warning("Ignoring malformed image GPS metadata: %s", exc)
                gps = {}
            return ImageOps.exif_transpose(image).convert("RGB"), _coordinates_from_gps_ifd(gps)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded image could not be decoded.") from exc


def _extract_video_gps(path: Path) -> dict[str, float] | None:
    try:
        with av.open(str(path)) as container:
            metadata_groups = [container.metadata]
            metadata_groups.extend(stream.metadata for stream in container.streams)
            return _coordinates_from_video_metadata(metadata_groups)
    except (av.error.FFmpegError, OSError) as exc:
        _LOGGER.warning("Could not read embedded GPS metadata from the uploaded video: %s", exc)
        return None


def _load_video_frames(path: Path) -> list[tuple[int, float | None, Image.Image]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("The uploaded video could not be opened.")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = fps if math.isfinite(fps) and fps > 0 else 0
        if frame_count > 0:
            indices = np.linspace(
                0,
                frame_count - 1,
                min(frame_count, MAX_VIDEO_FRAMES),
                dtype=int,
            )
            frames: list[tuple[int, float | None, Image.Image]] = []
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if ok:
                    timestamp = float(index / fps) if fps else None
                    frames.append(
                        (
                            int(index),
                            timestamp,
                            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                        )
                    )
        else:
            frames = []
            index = 0
            while len(frames) < MAX_VIDEO_FRAMES:
                ok, frame = capture.read()
                if not ok:
                    break
                if index % 30 == 0:
                    timestamp = float(index / fps) if fps else None
                    frames.append(
                        (
                            index,
                            timestamp,
                            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                        )
                    )
                index += 1
    finally:
        capture.release()

    if not frames:
        raise ValueError("No readable frames were found in the uploaded video.")
    return frames


def _normalized_bbox(bbox: BBox) -> dict[str, float] | None:
    x_min, y_min, x_max, y_max = (
        float(bbox.x_min),
        float(bbox.y_min),
        float(bbox.x_max),
        float(bbox.y_max),
    )
    if not all(math.isfinite(value) for value in (x_min, y_min, x_max, y_max)):
        return None
    normalized = {
        "x_min": max(0.0, min(1.0, x_min)),
        "y_min": max(0.0, min(1.0, y_min)),
        "x_max": max(0.0, min(1.0, x_max)),
        "y_max": max(0.0, min(1.0, y_max)),
    }
    if normalized["x_min"] >= normalized["x_max"] or normalized["y_min"] >= normalized["y_max"]:
        return None
    return normalized


def _intersection_over_union(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    intersection_width = max(
        0.0, min(left["x_max"], right["x_max"]) - max(left["x_min"], right["x_min"])
    )
    intersection_height = max(
        0.0, min(left["y_max"], right["y_max"]) - max(left["y_min"], right["y_min"])
    )
    intersection = intersection_width * intersection_height
    left_area = (left["x_max"] - left["x_min"]) * (left["y_max"] - left["y_min"])
    right_area = (right["x_max"] - right["x_min"]) * (right["y_max"] - right["y_min"])
    return intersection / (left_area + right_area - intersection)


def _suppress_duplicate_boxes(
    detections: list[tuple[Detection, dict[str, float]]],
) -> list[tuple[Detection, dict[str, float]]]:
    candidates = sorted(detections, key=lambda item: float(item[0].confidence), reverse=True)
    selected: list[tuple[Detection, dict[str, float]]] = []
    for candidate in candidates:
        if all(
            _intersection_over_union(candidate[1], previous[1]) <= DETECTOR_NMS_IOU_THRESHOLD
            for previous in selected
        ):
            selected.append(candidate)
    return selected


def _classify_frame_detections(
    frame: tuple[int, float | None, Image.Image],
    detector_result: DetectResult,
) -> list[dict[str, object]]:
    frame_index, timestamp, image = frame
    crops: list[Image.Image] = []
    records: list[dict[str, object]] = []
    animal_boxes: list[tuple[Detection, dict[str, float]]] = []
    for detection in detector_result.detections:
        if detection.label.casefold() != "animal":
            continue
        detector_confidence = float(detection.confidence)
        if not math.isfinite(detector_confidence) or detector_confidence < DETECTOR_MIN_CONFIDENCE:
            continue
        bbox = _normalized_bbox(detection.bbox)
        if bbox is None:
            _LOGGER.warning("Ignoring an invalid animal bounding box in frame %s.", frame_index)
            continue
        animal_boxes.append((detection, bbox))

    # RT-DETR top-K output can contain overlapping proposals for one animal.
    for detection, bbox in _suppress_duplicate_boxes(animal_boxes):
        left = max(0, min(image.width, math.floor(bbox["x_min"] * image.width)))
        top = max(0, min(image.height, math.floor(bbox["y_min"] * image.height)))
        right = max(0, min(image.width, math.ceil(bbox["x_max"] * image.width)))
        bottom = max(0, min(image.height, math.ceil(bbox["y_max"] * image.height)))
        if right <= left or bottom <= top:
            _LOGGER.warning("Ignoring an empty animal crop in frame %s.", frame_index)
            continue
        crops.append(image.crop((left, top, right, bottom)))
        records.append(
            {
                "frame_index": frame_index,
                "timestamp_seconds": timestamp,
                "bbox": bbox,
                "detector_confidence": float(detection.confidence),
            }
        )

    if not crops:
        return []
    probabilities = _predict(crops)
    for record, scores in zip(records, probabilities):
        class_index = int(scores.argmax().item())
        record["species"] = _display_name(_CLASS_NAMES[class_index])
        record["confidence"] = float(scores[class_index].item())
        record["needs_review"] = record["confidence"] < MIN_CONFIDENCE
    return records


def _ranked_candidates(detections: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[float]] = {}
    for detection in detections:
        confidence = float(detection["confidence"])
        if confidence >= MIN_CONFIDENCE:
            grouped.setdefault(str(detection["species"]), []).append(confidence)
    ranked = [
        {
            "species": species,
            "scientific": "",
            "confidence": float(np.mean(scores)),
            "count": len(scores),
        }
        for species, scores in grouped.items()
    ]
    return sorted(ranked, key=lambda item: (-int(item["count"]), -float(item["confidence"])))


def _detect_frames(
    frames: list[tuple[int, float | None, Image.Image]],
    image_paths: list[str],
) -> list[dict[str, object]]:
    with _INFERENCE_LOCK:
        detector_results = sparrow_engine.detect(
            image_paths,
            model=DETECTOR_MODEL_ID,
            threshold=DETECTOR_MIN_CONFIDENCE,
            max_detections=MAX_DETECTIONS_PER_FRAME,
        )
        if len(detector_results) != len(frames):
            raise RuntimeError("The wildlife detector returned an incomplete frame result.")
        return [
            record
            for frame, detector_result in zip(frames, detector_results)
            for record in _classify_frame_detections(frame, detector_result)
        ]


def _run_inference(
    path: Path, media_type: str
) -> dict[str, object]:
    if media_type == "image":
        image, location = _load_image(path)
        frames = [(0, None, image)]
        location_source = "image EXIF GPS"
        with tempfile.TemporaryDirectory(prefix="wildguard-image-") as directory:
            detector_path = Path(directory) / "image.jpg"
            image.save(detector_path, format="JPEG", quality=95)
            detections = _detect_frames(frames, [str(detector_path)])
    else:
        frames = _load_video_frames(path)
        location = _extract_video_gps(path)
        location_source = "video metadata"
        with tempfile.TemporaryDirectory(prefix="wildguard-frames-") as directory:
            image_paths = []
            for frame_index, _, image in frames:
                frame_path = Path(directory) / f"frame-{frame_index:06d}.jpg"
                image.save(frame_path, format="JPEG", quality=90)
                image_paths.append(str(frame_path))
            detections = _detect_frames(frames, image_paths)
    candidates = _ranked_candidates(detections)
    return {
        "predictions": candidates,
        "candidates": candidates,
        "detections": detections,
        "sampled_frames": len(frames),
        "image_size": [frames[0][2].width, frames[0][2].height],
        "location": (
            {**location, "source": location_source}
            if location is not None
            else None
        ),
    }


def _connect_database() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def _record_analysis(name: str, media_type: str, result: dict[str, object]) -> None:
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
                    name,
                    media_type,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    json.dumps(result["predictions"]),
                    json.dumps(result["candidates"]),
                    result["sampled_frames"],
                    json.dumps(result["detections"]),
                    location.get("latitude"),
                    location.get("longitude"),
                    location.get("source"),
                ),
            )


@app.get("/api/health")
def health() -> dict[str, object]:
    model_loaded = _MODEL is not None
    detector_ready = _DETECTOR_READY
    ready = model_loaded and detector_ready
    return {
        "status": "ok" if ready else "not_ready",
        "capabilities": {
            "detect": detector_ready,
            "species": model_loaded,
            "audio": False,
            "assistant": False,
        },
        "message": (
            "Animal detector and crop classifier are configured."
            if ready
            else "Wildlife detector or classifier is not loaded."
        ),
        "classes": len(_CLASS_NAMES),
        "minimum_confidence": MIN_CONFIDENCE,
    }


@app.get("/api/dashboard")
def dashboard() -> dict[str, object]:
    with closing(_connect_database()) as connection:
        rows = connection.execute(
            "SELECT created_at, predictions_json, detections_json FROM analyses ORDER BY created_at"
        ).fetchall()

    total_predictions = 0
    species: set[str] = set()
    confidences: list[float] = []
    daily: dict[str, int] = {}
    for row in rows:
        predictions = json.loads(row["predictions_json"])
        detections = json.loads(row["detections_json"] or "[]")
        day = row["created_at"][:10]
        if detections:
            counted = [
                detection
                for detection in detections
                if float(detection.get("confidence", 0)) >= MIN_CONFIDENCE
            ]
            total_predictions += len(counted)
            species.update(str(detection["species"]) for detection in counted)
            confidences.extend(float(detection["confidence"]) for detection in counted)
            daily[day] = daily.get(day, 0) + len(counted)
        else:
            total_predictions += len(predictions)
            species.update(str(prediction["species"]) for prediction in predictions)
            confidences.extend(float(prediction["confidence"]) for prediction in predictions)
            daily[day] = daily.get(day, 0) + len(predictions)

    return {
        "total_predictions": total_predictions,
        "species_identified": len(species),
        "analysis_sessions": len(rows),
        "average_confidence": float(np.mean(confidences)) if confidences else None,
        "daily": [{"date": date, "count": count} for date, count in sorted(daily.items())],
    }


@app.get("/api/history")
def history(limit: int = Query(default=6, ge=1, le=100)) -> list[dict[str, object]]:
    with closing(_connect_database()) as connection:
        rows = connection.execute(
            "SELECT name, media_type, predictions_json, candidates_json "
            "FROM analyses ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

    entries = []
    for row in rows:
        predictions = json.loads(row["predictions_json"])
        candidates = json.loads(row["candidates_json"])
        top = predictions[0] if predictions else (candidates[0] if candidates else {})
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
    suffix = Path(filename).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        media_type = "image"
    elif suffix in VIDEO_EXTENSIONS:
        media_type = "video"
    else:
        await file.close()
        raise HTTPException(status_code=415, detail="Upload a supported image or video file.")

    temp_path: Path | None = None
    try:
        if (browser_latitude is None) != (browser_longitude is None):
            raise HTTPException(status_code=422, detail="Both browser location coordinates are required.")
        browser_location = (
            _valid_coordinates(browser_latitude, browser_longitude)
            if browser_latitude is not None and browser_longitude is not None
            else None
        )
        if browser_latitude is not None and browser_location is None:
            raise HTTPException(status_code=422, detail="The browser location coordinates are invalid.")

        size = 0
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            temp_path = Path(temporary.name)
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="The upload exceeds the 200 MB limit.")
                temporary.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        result = await asyncio.to_thread(_run_inference, temp_path, media_type)
        result["location"] = (
            {**browser_location, "source": "browser upload location"}
            if browser_location is not None
            else None
        )
        if not result["detections"]:
            note = (
                "No animals were detected in the full image or sampled video frames."
            )
        elif media_type == "video":
            note = (
                f"Detected {len(result['detections'])} animal instances across "
                f"{result['sampled_frames']} sampled frames. Species scores below 50% need review."
            )
        else:
            note = (
                f"Detected {len(result['detections'])} animal instance(s) and classified each crop. "
                "Species scores below 50% need review."
            )
        if browser_location is None:
            note += " Browser location was unavailable or denied; no map spot was recorded."
        else:
            note += " The map spot marks the browser's upload location, not the animal's exact position."

        result.update(
            {
                "media_type": media_type,
                "note": note,
            }
        )
        _record_analysis(filename[:255], media_type, result)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except sparrow_engine.SparrowEngineError as exc:
        raise HTTPException(status_code=503, detail=f"Wildlife detector error: {exc}") from exc
    finally:
        await file.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_FILES["index.html"])


@app.get("/{asset_path:path}")
def static_asset(asset_path: str) -> FileResponse:
    asset = STATIC_FILES.get(asset_path)
    if asset is not None:
        return FileResponse(asset)

    parts = Path(asset_path).parts
    if len(parts) == 2 and parts[0] in STATIC_DIRECTORIES:
        directory, filename = parts
        if Path(filename).suffix.lower() in STATIC_EXTENSIONS:
            asset_root = FRONTEND_ROOT / directory if directory == "images" else ROOT / directory
            asset = (asset_root / filename).resolve()
            if asset.parent == asset_root.resolve() and asset.is_file():
                return FileResponse(asset)

    raise HTTPException(status_code=404, detail="File not found.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8001")),
    )
