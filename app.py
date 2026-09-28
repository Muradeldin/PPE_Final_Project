import uvicorn
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pipeline_ncnn_web import EdgePPEPipeline, MODEL_PATH

# Video file for testing; on the Pi use 0 for the first camera
SOURCE = "media/cctv_test.mp4"

app = FastAPI(title="Edge PPE Controller")
pipeline_instance = None

# Let the control page call this API even when it's opened from another address or as a local file
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/start")
def start_detection(background_tasks: BackgroundTasks):
    global pipeline_instance

    # Prevent starting multiple instances and crashing the Pi
    if pipeline_instance and pipeline_instance.running:
        return {"status": "Pipeline is already running"}

    # Initialize the pipeline from your imported file
    pipeline_instance = EdgePPEPipeline(
        source=SOURCE,
        model_path=MODEL_PATH
    )
    # Mark as running now, so /status and repeated clicks see it before the thread starts
    pipeline_instance.running = True

    # Run the pipeline loop in a background thread
    background_tasks.add_task(pipeline_instance.run)
    return {"status": "Detection Started"}

@app.post("/stop")
def stop_detection():
    global pipeline_instance
    if pipeline_instance:
        pipeline_instance.running = False
        return {"status": "Stopping detection..."}
    return {"status": "No pipeline is currently running"}

@app.get("/status")
def status():
    """Polled by the control page every few seconds."""
    running = bool(pipeline_instance and pipeline_instance.running)
    return {"running": running, "source": str(SOURCE)}

# Control page (control/index.html), served at http://<pi-ip>:8000/
# Mounted last so the API routes above take priority
app.mount("/", StaticFiles(directory=Path(__file__).parent / "control", html=True), name="control")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
