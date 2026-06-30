from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from cardiac_image_system.core.io import load_grayscale_image, save_grayscale_image
from cardiac_image_system.core.preprocessing import preprocess_image

APP_ROOT = Path("runtime")
UPLOAD_DIR = APP_ROOT / "uploads"
PROCESSED_DIR = APP_ROOT / "processed"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="AP25794129 Cardio Preprocessing Prototype",
    description="Research prototype. Not clinical diagnostic software.",
    version="0.1.0",
)


class PreprocessRequest(BaseModel):
    file_id: str
    mode: str = "hybrid"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "scope": "research prototype only"}


@app.post("/upload")
def upload_image(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    file_id = str(uuid.uuid4())
    saved_path = UPLOAD_DIR / f"{file_id}{suffix}"

    with saved_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "file_id": file_id,
        "original_filename": file.filename,
        "saved_path": str(saved_path),
        "warning": "Do not upload identifiable patient data.",
    }


@app.post("/preprocess")
def preprocess(req: PreprocessRequest) -> dict:
    candidates = list(UPLOAD_DIR.glob(f"{req.file_id}.*"))
    if not candidates:
        raise HTTPException(status_code=404, detail="file_id not found")

    try:
        image = load_grayscale_image(candidates[0])
        processed = preprocess_image(image, mode=req.mode)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    output_path = PROCESSED_DIR / f"{req.file_id}_{req.mode}.png"
    save_grayscale_image(output_path, processed)

    return {
        "file_id": req.file_id,
        "mode": req.mode,
        "processed_path": str(output_path),
        "scope": "research preprocessing result; not diagnostic output",
    }
