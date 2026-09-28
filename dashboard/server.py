"""
PPE Violation Dashboard.

Receives violation alerts from the Pi (pipeline_ncnn_web.py) and shows them on a web page.

Run on the Mac:   python dashboard/server.py
Then open:        http://localhost:8080  (or http://<Mac Tailscale IP>:8080 from another device)
"""
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ==============================================================================
# CONFIGURATION
# ==============================================================================

HOST = "0.0.0.0"   # Listen on all interfaces so the Pi can reach it over Tailscale
PORT = 8080        # Must match the port in CLOUD_WEBHOOK_URL (pipeline_ncnn_web.py)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "violations"
DB_PATH = DATA_DIR / "violations.db"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# DATABASE
# ==============================================================================

@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


with db() as conn:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            received_at  TEXT    NOT NULL,
            worker_id    TEXT    NOT NULL,
            reasons      TEXT    NOT NULL,
            image        TEXT    NOT NULL,
            acknowledged INTEGER NOT NULL DEFAULT 0
        )
    """)

# ==============================================================================
# API
# ==============================================================================

app = FastAPI(title="PPE Violation Dashboard")


@app.post("/api/upload-crop")
async def upload_crop(
    file: UploadFile = File(...),
    worker_id: str = Form(...),
    reasons: str = Form(...),
):
    """Called by the Pi for every violation: saves the crop and records the alert."""
    now = datetime.now()
    image_name = f"{now:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}.jpg"
    (IMAGES_DIR / image_name).write_bytes(await file.read())

    with db() as conn:
        conn.execute(
            "INSERT INTO violations (received_at, worker_id, reasons, image) VALUES (?, ?, ?, ?)",
            (now.isoformat(timespec="seconds"), worker_id, reasons, image_name),
        )

    print(f"[ALERT] Worker {worker_id}: {reasons}")
    return {"status": "ok"}


@app.get("/api/violations")
def list_violations(limit: int = 200):
    with db() as conn:
        rows = conn.execute("SELECT * FROM violations ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


@app.post("/api/violations/{violation_id}/acknowledge")
def acknowledge(violation_id: int):
    with db() as conn:
        updated = conn.execute(
            "UPDATE violations SET acknowledged = 1 WHERE id = ?", (violation_id,)
        ).rowcount
    if updated == 0:
        raise HTTPException(status_code=404, detail="Violation not found")
    return {"status": "ok"}


@app.post("/api/violations/acknowledge-all")
def acknowledge_all():
    with db() as conn:
        conn.execute("UPDATE violations SET acknowledged = 1 WHERE acknowledged = 0")
    return {"status": "ok"}


app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "index.html")


if __name__ == "__main__":
    print(f"[INFO] Dashboard running on http://localhost:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
