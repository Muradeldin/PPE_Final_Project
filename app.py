"""
Pi main program: lets the website start/stop detection through Supabase.

Every SYNC_SECONDS it reads the Start/Stop state the website set (device_status.desired_running),
starts or stops the pipeline to match, and reports back whether it's running, its FPS and source.
The Pi only makes outgoing connections, so it works behind any router.

Run on the Pi:  python app.py      (Ctrl+C to quit)
"""
import threading
import time

import cloud
from pipeline_ncnn_web import EdgePPEPipeline, MODEL_PATH

# Video file for testing; on the Pi use 0 for the first camera
SOURCE = "media/cctv_test.mp4"
SYNC_SECONDS = 2.0

pipeline_instance = None
pipeline_thread = None


def start_pipeline():
    global pipeline_instance, pipeline_thread
    print(f"[AGENT] Starting detection on {SOURCE}")
    pipeline_instance = EdgePPEPipeline(source=SOURCE, model_path=MODEL_PATH)
    pipeline_instance.running = True  # Mark now so the next sync sees it before the thread starts
    pipeline_thread = threading.Thread(target=pipeline_instance.run, daemon=True)
    pipeline_thread.start()


def stop_pipeline():
    print("[AGENT] Stopping detection")
    pipeline_instance.running = False
    pipeline_thread.join(timeout=10)


def is_running():
    return bool(pipeline_instance and pipeline_instance.running)


def main():
    was_running = False
    print(f"[AGENT] Connected to {cloud.SUPABASE_URL}, waiting for Start from the website...")

    while True:
        try:
            desired = cloud.get_desired_running()
            running = is_running()

            if was_running and not running and desired:
                # Stopped by itself (video ended or camera lost): reset the button instead of restarting
                print("[AGENT] Detection ended on its own")
                cloud.update_device_status(desired_running=False, is_running=False)
                desired = False
            elif desired and not running:
                start_pipeline()
            elif not desired and running:
                stop_pipeline()

            running = is_running()
            fps = round(pipeline_instance.fps, 1) if running else None
            cloud.update_device_status(is_running=running, fps=fps, source=str(SOURCE))
            was_running = running
        except Exception as e:
            # Network hiccups shouldn't kill the agent; detection keeps running meanwhile
            print(f"[AGENT] Sync with Supabase failed: {e}")

        time.sleep(SYNC_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        if is_running():
            stop_pipeline()
        try:
            cloud.update_device_status(is_running=False, fps=None)
        except Exception:
            pass
        print("[AGENT] Bye")
