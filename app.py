import uvicorn
from fastapi import FastAPI, BackgroundTasks
from pipline_ncnn_web import EdgePPEPipeline

app = FastAPI(title="Edge PPE Controller")
pipeline_instance = None

@app.post("/start")
def start_detection(background_tasks: BackgroundTasks):
    global pipeline_instance
    
    # Prevent starting multiple instances and crashing the Pi
    if pipeline_instance and pipeline_instance.running:
        return {"status": "Pipeline is already running"}
    
    # Initialize the pipeline from your imported file
    pipeline_instance = EdgePPEPipeline(
        source="media/cctv_test.mp4", 
        model_path="best_yolo8_ncnn_model"
    )
    
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

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)