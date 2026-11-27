"""Lightweight API for deployments that cannot host the AI inference models."""

import json
import os
import sqlite3
from contextlib import asynccontextmanager, closing
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent
DATABASE_PATH = Path(
    os.getenv("DATABASE_PATH", str(ROOT / "data" / "wildguard.sqlite3"))
).resolve()


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
    _initialize_database()
    yield


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


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "capabilities": {
            "detect": False,
            "species": False,
            "audio": False,
            "assistant": False,
        },
        "message": "Lightweight API is online; on-host AI inference is disabled.",
        "classes": 0,
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
def analyze() -> None:
    raise HTTPException(
        status_code=503,
        detail=(
            "AI analysis is disabled on this lightweight deployment. "
            "Use the full backend or connect an external inference service."
        ),
    )
