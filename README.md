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

In `app.py`, change:

```python
SOURCE = "media/cctv_test.mp4"
```

to

```python
SOURCE = 0
```

This works for a **USB webcam**. The **Raspberry Pi Camera Module** (ribbon cable) is not supported
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

### Updating the Pi later

```bash
cd PPE_Final_Project
git pull
sudo systemctl restart ppe
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
