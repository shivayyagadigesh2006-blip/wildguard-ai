"""Lightweight API for deployments that cannot host the AI inference models."""

import json
import logging
import os
import sqlite3
from contextlib import asynccontextmanager, closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, BinaryIO, Mapping

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent
DATABASE_PATH = Path(
    os.getenv("DATABASE_PATH", str(ROOT / "data" / "wildguard.sqlite3"))
).resolve()
INFERENCE_API_URL = os.getenv("INFERENCE_API_URL", "").rstrip("/")
INFERENCE_API_KEY = os.getenv("INFERENCE_API_KEY", "")
_HTTP_CLIENT: httpx.AsyncClient | None = None
_LOGGER = logging.getLogger(__name__)


def _connect_database() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def _initialize_database() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect_database()) as connection:
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
        connection.commit()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    global _HTTP_CLIENT
    _initialize_database()
    if INFERENCE_API_URL and not INFERENCE_API_KEY:
        raise RuntimeError("INFERENCE_API_KEY is required when INFERENCE_API_URL is configured.")
    if INFERENCE_API_URL:
        _HTTP_CLIENT = httpx.AsyncClient(headers={"X-API-Key": INFERENCE_API_KEY})
    try:
        yield
    finally:
        if _HTTP_CLIENT is not None:
            await _HTTP_CLIENT.aclose()
            _HTTP_CLIENT = None


app = FastAPI(title="WildGuard AI Lightweight API", version="1.0.0", lifespan=lifespan)

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


async def _request_inference(
    method: str,
    path: str,
    *,
    params: Mapping[str, str | int] | None = None,
    files: Mapping[str, tuple[str, BinaryIO, str]] | None = None,
    data: Mapping[str, str] | None = None,
    timeout: float | httpx.Timeout = 30.0,
) -> Any:
    if _HTTP_CLIENT is None:
        raise HTTPException(
            status_code=503,
            detail="External AI inference is not configured on this deployment.",
        )

    try:
        response = await _HTTP_CLIENT.request(
            method,
            f"{INFERENCE_API_URL}{path}",
            params=params,
            files=files,
            data=data,
            timeout=timeout,
        )
    except httpx.TimeoutException as exc:
        _LOGGER.warning("External AI inference timed out on %s.", path)
        raise HTTPException(status_code=504, detail="External AI inference timed out.") from exc
    except httpx.RequestError as exc:
        _LOGGER.error("External AI inference request failed on %s: %s", path, exc)
        raise HTTPException(
            status_code=502,
            detail="External AI inference service is unavailable.",
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        _LOGGER.error("External AI inference returned invalid JSON on %s.", path)
        raise HTTPException(
            status_code=502,
            detail="External AI inference service returned an invalid response.",
        ) from exc
    if response.is_error:
        detail = payload.get("detail", "External AI inference request failed.") if isinstance(payload, dict) else payload
        raise HTTPException(status_code=response.status_code, detail=detail)
    return payload


def _record_analysis(name: str, media_type: str, result: Mapping[str, Any]) -> None:
    location = result.get("location")
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
                    media_type,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    json.dumps(result.get("predictions", [])),
                    json.dumps(result.get("candidates", [])),
                    int(result.get("sampled_frames", 0)),
                    json.dumps(result.get("detections", [])),
                    location.get("latitude"),
                    location.get("longitude"),
                    location.get("source"),
                ),
            )


@app.get("/api/health")
def health() -> dict[str, object]:
    inference_configured = bool(INFERENCE_API_URL and INFERENCE_API_KEY)
    return {
        "status": "ok" if inference_configured else "not_ready",
        "capabilities": {
            "detect": inference_configured,
            "species": inference_configured,
            "audio": False,
            "assistant": False,
        },
        "message": (
            "External AI inference is configured."
            if inference_configured
            else "Configure external inference to enable wildlife analysis."
        ),
        "classes": 27 if inference_configured else 0,
        "minimum_confidence": 0.5,
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
        counted = (
            [
                detection
                for detection in detections
                if float(detection.get("confidence", 0)) >= 0.5
            ]
            if detections
            else predictions
        )
        total_predictions += len(counted)
        species.update(str(item["species"]) for item in counted)
        confidences.extend(float(item["confidence"]) for item in counted)
        daily[day] = daily.get(day, 0) + len(counted)

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
) -> dict[str, Any]:
    if not INFERENCE_API_URL:
        raise HTTPException(
            status_code=503,
            detail="Configure INFERENCE_API_URL to enable wildlife analysis.",
        )

    data: dict[str, str] = {}
    if browser_latitude is not None:
        data["browser_latitude"] = str(browser_latitude)
    if browser_longitude is not None:
        data["browser_longitude"] = str(browser_longitude)

    try:
        await file.seek(0)
        result = await _request_inference(
            "POST",
            "/api/analyze",
            files={
                "file": (
                    Path(file.filename or "upload").name,
                    file.file,
                    file.content_type or "application/octet-stream",
                )
            },
            data=data,
            timeout=httpx.Timeout(600.0, connect=15.0),
        )
    finally:
        await file.close()

    if not isinstance(result, dict):
        raise HTTPException(
            status_code=502,
            detail="External AI inference service returned an invalid analysis response.",
        )
    media_type = str(result.get("media_type", "unknown"))
    _record_analysis(
        Path(file.filename or "upload").name,
        media_type,
        result,
    )
    return result
