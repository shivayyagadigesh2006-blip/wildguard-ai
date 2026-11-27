"""Download the licensed MegaDetector model used by WildGuard AI."""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "models" / "sparrow"
MODEL_ID = "MDV6-apa-rtdetr-e"
MODEL_ZIP = "camera_trap__detector__MDV6-apa-rtdetr-e.zip"
ZENODO_RECORD = "23992557"
MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024


def _download(url: str, destination: Path, expected_size: int | None = None) -> str:
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl is not None:
        subprocess.run(
            [
                curl,
                "--fail",
                "--location",
                "--progress-bar",
                "--connect-timeout",
                "30",
                "--max-time",
                "1800",
                "--retry",
                "3",
                "--retry-delay",
                "2",
                "--output",
                str(destination),
                url,
            ],
            check=True,
        )
        size = destination.stat().st_size
        if size > MAX_DOWNLOAD_BYTES or (expected_size is not None and size > expected_size):
            raise ValueError("The detector download exceeded its published size.")
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "WildGuard-AI/1.0"})
        size = 0
        with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
            while chunk := response.read(64 * 1024):
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES or (expected_size is not None and size > expected_size):
                    raise ValueError("The detector download exceeded its published size.")
                output.write(chunk)
                output.flush()
    if size > MAX_DOWNLOAD_BYTES:
        raise ValueError("The detector download exceeded the allowed size.")
    if expected_size is not None and size != expected_size:
        raise ValueError("The detector download size does not match the Zenodo record.")
    digest = hashlib.md5(usedforsecurity=False)
    with destination.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> Path:
    root = destination.resolve()
    model_root = (root / MODEL_ID).resolve()
    total_size = 0
    for member in archive.infolist():
        member_path = (root / member.filename).resolve()
        if not member_path.is_relative_to(root) or not member_path.is_relative_to(model_root):
            raise ValueError("The detector archive contains an unexpected file path.")
        total_size += member.file_size
        if total_size > 1024 * 1024 * 1024:
            raise ValueError("The detector archive exceeds the allowed uncompressed size.")
    archive.extractall(root)
    if not (model_root / "manifest.toml").is_file() or not any(model_root.rglob("*.onnx")):
        raise ValueError("The detector archive is missing its manifest or ONNX model.")
    return model_root


def install_detector() -> Path:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    target = MODEL_ROOT / MODEL_ID
    if (target / "manifest.toml").is_file() and any(target.rglob("*.onnx")):
        print(f"Detector is already installed at {target}")
        return target
    if target.exists():
        raise RuntimeError(f"Incomplete detector directory exists at {target}; inspect it before retrying.")

    api_url = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
    request = urllib.request.Request(api_url, headers={"User-Agent": "WildGuard-AI/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        record = json.load(response)
    entry = next((item for item in record.get("files", []) if item.get("key") == MODEL_ZIP), None)
    if entry is None:
        raise RuntimeError(f"{MODEL_ZIP} is not listed in Zenodo record {ZENODO_RECORD}.")
    checksum = str(entry.get("checksum", ""))
    if not checksum.startswith("md5:") or not isinstance(entry.get("size"), int):
        raise RuntimeError("The Zenodo record does not provide the expected model checksum and size.")
    expected_md5 = checksum.removeprefix("md5:")
    expected_size = entry["size"]
    if expected_size <= 0 or expected_size > MAX_DOWNLOAD_BYTES:
        raise RuntimeError("The published detector archive size is outside the allowed limit.")

    zip_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".wildguard-detector-", suffix=".zip", dir=MODEL_ROOT, delete=False
        ) as temporary:
            zip_path = Path(temporary.name)
        url = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{MODEL_ZIP}?download=1"
        actual_md5 = _download(url, zip_path, expected_size)
        if actual_md5 != expected_md5:
            raise RuntimeError("The detector checksum does not match the Zenodo record.")

        with tempfile.TemporaryDirectory(prefix=".wildguard-stage-", dir=MODEL_ROOT) as staging:
            with zipfile.ZipFile(zip_path) as archive:
                staged_model = _safe_extract(archive, Path(staging))
            (staged_model / ".sparrow_zip_md5").write_text(
                expected_md5 + os.linesep, encoding="ascii"
            )
            os.replace(staged_model, target)
    finally:
        if zip_path is not None:
            zip_path.unlink(missing_ok=True)

    print(f"Installed {MODEL_ID} at {target}")
    return target


def main() -> None:
    try:
        install_detector()
    except (
        OSError,
        ValueError,
        urllib.error.URLError,
        zipfile.BadZipFile,
        RuntimeError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        raise SystemExit(f"Could not install the wildlife detector: {exc}") from exc


if __name__ == "__main__":
    main()
