# Edge PPE Monitor

A Raspberry Pi 4 watches a construction-site camera, detects workers without a **helmet** or **vest**
with a YOLOv8 model, and reports each violation (with a photo) to a website where the team can
start/stop detection and review alerts.

```
 Raspberry Pi (app.py)  ──uploads / heartbeat──►  Supabase  ◄──sign in, read, Start/Stop──  Website (Vercel)
   camera → YOLOv8 NCNN → violations                database + photos                    ppe-pi-dashboard.vercel.app
```

The Pi only makes outgoing connections, so it works behind any router with no port forwarding.

## What's in the repo

| Path | Purpose |
|---|---|
| `app.py` | **Run this on the Pi.** Follows the website's Start/Stop and reports status. |
| `pipeline_ncnn_web.py` | Detection pipeline that uploads violations to Supabase (used by `app.py`). |
| `pipeline_ncnn.py` | Same detection, but saves crops to `worker_crops/` and shows a video window. For testing on a PC. |
| `cloud.py` | Supabase connection for the Pi. |
| `stream_server.py` | Livestream of the detection video (used by `app.py`). |
| `models/best_yolo8_ncnn_model_half/` | The model the Pi uses (YOLOv8n, NCNN, 320 px). |
| `control/` | The website (static files, deployed to Vercel). |
| `supabase/schema.sql` | Database tables, photo storage and permissions (already applied). |
| `media/` | Test videos. |
| `legacy/` | Earlier experiments, not maintained. |

## Setting up the Raspberry Pi

Tested target: Raspberry Pi 4 with **Raspberry Pi OS 64-bit (Bookworm or newer)**. Check with
`uname -m`, which should print `aarch64`.

### 1. Install

```bash
sudo apt update
sudo apt install -y git python3-venv libgl1 libglib2.0-0

git clone https://github.com/Muradeldin/PPE_Final_Project.git
cd PPE_Final_Project

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The install downloads PyTorch and takes a while (10–20 minutes on a Pi 4).

### 2. Add the Supabase secret key

```bash
cp .env.example .env
nano .env
```

Replace `sb_secret_paste_yours_here` with the project's **secret key** (Supabase → Project Settings →
API Keys). Get it from Adham privately. **Never commit `.env` or post the key anywhere.**
`.env` is already in `.gitignore`.

### 3. Test with the sample video

```bash
source .venv/bin/activate
python app.py
```

It should print `Connected to https://…supabase.co, waiting for Start from the website...`.
Open https://ppe-pi-dashboard.vercel.app, sign in, and press **Start detection**. The Pi prints
its FPS every 2 seconds, and violations appear on the website within about 10 seconds. When the
video ends, the website goes back to Offline by itself. Stop the program with `Ctrl+C`.

### 4. Switch to the camera

Add this line to `.env`:

```bash
SOURCE=0
```

This works for a **USB webcam** (`0` is the first camera). Without it, the test video is used. The **Raspberry Pi Camera Module** (ribbon cable) is not supported
by this code yet; it needs a small change to read frames with `picamera2`.

### 5. Start automatically on boot

Run this from inside the `PPE_Final_Project` folder. It fills in your username and folder automatically:

```bash
sudo tee /etc/systemd/system/ppe.service > /dev/null <<EOF
[Unit]
Description=Edge PPE Monitor
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$PWD
ExecStart=$PWD/.venv/bin/python app.py
Environment=PYTHONUNBUFFERED=1
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now ppe
```

Useful commands:

```bash
journalctl -u ppe -f          # watch the logs live
sudo systemctl restart ppe    # restart after changing the code
sudo systemctl stop ppe       # stop it
```

### 6. Livestream (Tailscale Funnel)

The website's **Live view** shows the camera with the detection boxes while detection runs.
`app.py` serves it on port 8000 of the Pi; Tailscale Funnel publishes it over HTTPS.

1. Funnel must be enabled once in the Pi's Tailscale network (run `sudo tailscale funnel 8000`, open
   the link it prints with the tailnet owner's account, click **Enable**, then `Ctrl+C`).
2. Publish port 8000 (stays on across reboots):
   ```bash
   sudo tailscale funnel --bg 8000
   sudo tailscale funnel status      # shows the public https://<pi-name>.<tailnet>.ts.net address
   ```
3. Put that address in `.env`:
   ```bash
   STREAM_PUBLIC_URL=https://muradppe.tail569fb8.ts.net
   ```
4. Restart: `sudo systemctl restart ppe`

The stream needs a secret token that changes every time `app.py` starts; the website gets the full
link from Supabase after sign-in, so strangers who find the address get `403 Forbidden`. It only
uses bandwidth (about 2–3 Mbit/s per viewer) while someone has the Live view open.

### Alternative: run with Docker instead of steps 1 and 5

Use **either** Docker **or** the systemd service, never both (two copies would fight over the camera).
If the service is installed, turn it off first: `sudo systemctl disable --now ppe`.

```bash
cp .env.example .env && nano .env     # secret key, SOURCE=0, STREAM_PUBLIC_URL
docker compose up -d --build          # build and start (restarts automatically)
docker compose logs -f                # watch the logs
```

`docker-compose.yaml` passes the USB camera (`/dev/video0`) into the container and publishes the
livestream on the Pi's `127.0.0.1:8000`, so `sudo tailscale funnel --bg 8000` works the same way.
If no camera is plugged in, remove the `devices:` lines or the container won't start.

### Updating the Pi later

```bash
cd PPE_Final_Project
git pull
sudo systemctl restart ppe            # or with Docker: docker compose up -d --build
```

## Running on a PC (for testing)

```bash
pip install -r requirements.txt
python pipeline_ncnn.py        # detection with a video window, saves crops to worker_crops/
python app.py                  # same as the Pi: controlled from the website (needs .env)
```

## Website

The site in `control/` is plain HTML/CSS/JS that talks to Supabase directly. To deploy changes:

```bash
cd control
npx vercel --prod
```

Accounts are created in Supabase (Authentication → Users). Public sign-up is turned off.
