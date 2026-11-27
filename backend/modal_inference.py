"""Modal deployment for WildGuard's full wildlife inference API."""

import hmac
import os
import sys

import modal
from fastapi import Request
from fastapi.responses import JSONResponse

app = modal.App("wildguard-inference")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("backend/requirements-full.txt")
    .run_commands("mkdir -p /root/backend/models /root/backend/data")
    .add_local_file(
        "backend/full_app.py",
        remote_path="/root/backend/full_app.py",
        copy=True,
    )
    .add_local_file(
        "backend/download_detector.py",
        remote_path="/root/backend/download_detector.py",
        copy=True,
    )
    .add_local_file(
        "backend/models/best_wild_animal_classifier_b3.pth",
        remote_path="/root/backend/models/best_wild_animal_classifier_b3.pth",
        copy=True,
    )
    .add_local_dir("frontend", remote_path="/root/frontend", copy=True)
    .run_commands("cd /root/backend && python download_detector.py")
)


@app.function(
    image=image,
    cpu=4,
    memory=4096,
    timeout=900,
    secrets=[modal.Secret.from_name("wildguard-inference-key")],
)
@modal.asgi_app()
def inference_api():
    sys.path.insert(0, "/root/backend")
    from full_app import app as fastapi_app

    api_key = os.getenv("INFERENCE_API_KEY", "")
    if not api_key:
        raise RuntimeError("The Modal secret must define INFERENCE_API_KEY.")

    @fastapi_app.middleware("http")
    async def verify_api_key(request: Request, call_next):
        supplied_key = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(supplied_key, api_key):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid inference API key."},
            )
        return await call_next(request)

    return fastapi_app
