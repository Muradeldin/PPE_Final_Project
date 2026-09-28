"""
Livestream of the annotated detection video (MJPEG over HTTP).

Serves http://127.0.0.1:<port>/stream?token=<token>. On the Pi, Tailscale Funnel publishes it
over HTTPS (`sudo tailscale funnel --bg 8000`). The token keeps strangers out: the website only
gets the full link from Supabase after signing in.

Frames are only drawn and encoded while someone is watching, so an unwatched stream costs nothing.
"""
import hmac
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

# 480x360 at quality 60 is ~35 KB per frame: ~2.8 Mbit/s per viewer at 10 FPS
STREAM_SIZE = (480, 360)
JPEG_QUALITY = 60
MAX_STREAM_FPS = 10


def _placeholder_jpeg(text):
    """Dark card with a message, in the website's colours (BGR)."""
    w, h = STREAM_SIZE
    img = np.full((h, w, 3), (32, 24, 16), dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (104, 239, 197), 2)
    return cv2.imencode(".jpg", img)[1].tobytes()


class StreamServer:
    def __init__(self, get_pipeline, token, host="127.0.0.1", port=8000):
        self.get_pipeline = get_pipeline  # Returns the current pipeline (or None)
        self.token = token
        self.stopped_jpeg = _placeholder_jpeg("Detection is stopped")
        self.starting_jpeg = _placeholder_jpeg("Starting detection...")

        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                server._handle(self)

            def log_message(self, *args):
                pass  # Keep the console quiet

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True

    def start(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def _handle(self, request):
        url = urlparse(request.path)
        if url.path != "/stream":
            request.send_error(404)
            return
        token = parse_qs(url.query).get("token", [""])[0]
        if not hmac.compare_digest(token, self.token):
            request.send_error(403)
            return

        request.send_response(200)
        request.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        request.send_header("Cache-Control", "no-store")
        request.end_headers()

        last_frame_id = None
        last_sent = 0.0
        try:
            while True:
                pipeline = self.get_pipeline()
                running = pipeline is not None and pipeline.running
                idle = time.time() - last_sent

                if running:
                    pipeline.request_stream()  # Tells the pipeline to keep drawing frames
                    frame = pipeline.latest_frame
                    if frame is not None and pipeline.frame_id != last_frame_id:
                        last_frame_id = pipeline.frame_id
                        small = cv2.resize(frame, STREAM_SIZE, interpolation=cv2.INTER_AREA)
                        jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()
                        self._send(request, jpeg)
                        last_sent = time.time()
                    elif idle >= 3.0:
                        self._send(request, self.starting_jpeg)
                        last_sent = time.time()
                elif idle >= 1.0:
                    last_frame_id = None
                    self._send(request, self.stopped_jpeg)
                    last_sent = time.time()

                time.sleep(1.0 / MAX_STREAM_FPS)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # Viewer closed the page

    @staticmethod
    def _send(request, jpeg):
        request.wfile.write(
            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
            + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
        )
        request.wfile.flush()
