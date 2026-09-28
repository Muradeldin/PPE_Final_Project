"""
Talks to Supabase for the Pi: uploads violations and syncs the Start/Stop state with the website.

Needs SUPABASE_URL and SUPABASE_SECRET_KEY, either as environment variables or in a
.env file next to this script (see .env.example). The secret key must never go in git
or on the website.
"""
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

DEVICE_ID = "pi"              # Row in the device_status table
PHOTO_BUCKET = "violations"   # Supabase Storage bucket for violation photos
TIMEOUT = 10


def _load_env_file():
    """Minimal .env reader (KEY=value per line), so no extra package is needed."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file()
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SECRET_KEY = os.environ.get("SUPABASE_SECRET_KEY", "")

if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
    raise RuntimeError("Set SUPABASE_URL and SUPABASE_SECRET_KEY in .env (copy .env.example)")


def _headers(**extra):
    return {"apikey": SUPABASE_SECRET_KEY, "Authorization": f"Bearer {SUPABASE_SECRET_KEY}", **extra}


def upload_violation(worker_id, reasons, jpeg_bytes):
    """Stores the photo in the private bucket, then records the violation."""
    now = datetime.now(timezone.utc)
    image_path = f"{now:%Y/%m/%d}/{now:%H%M%S}_worker{worker_id}_{uuid.uuid4().hex[:6]}.jpg"

    response = requests.post(
        f"{SUPABASE_URL}/storage/v1/object/{PHOTO_BUCKET}/{image_path}",
        headers=_headers(**{"Content-Type": "image/jpeg"}),
        data=jpeg_bytes,
        timeout=TIMEOUT,
    )
    response.raise_for_status()

    response = requests.post(
        f"{SUPABASE_URL}/rest/v1/violations",
        headers=_headers(Prefer="return=minimal"),
        json={"worker_id": str(worker_id), "reasons": reasons, "image_path": image_path},
        timeout=TIMEOUT,
    )
    response.raise_for_status()


def get_desired_running():
    """What the website's Start/Stop buttons asked for."""
    response = requests.get(
        f"{SUPABASE_URL}/rest/v1/device_status",
        headers=_headers(),
        params={"id": f"eq.{DEVICE_ID}", "select": "desired_running"},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    rows = response.json()
    return bool(rows and rows[0]["desired_running"])


def update_device_status(**fields):
    """Heartbeat: reports is_running / fps / source etc. and stamps last_seen."""
    fields["last_seen"] = datetime.now(timezone.utc).isoformat()
    response = requests.patch(
        f"{SUPABASE_URL}/rest/v1/device_status",
        headers=_headers(Prefer="return=minimal"),
        params={"id": f"eq.{DEVICE_ID}"},
        json=fields,
        timeout=TIMEOUT,
    )
    response.raise_for_status()
