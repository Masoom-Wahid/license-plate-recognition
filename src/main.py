from fastapi import FastAPI, Request, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
from src.model import predict_and_save, process_video, output_dir_video
import asyncio
import shutil
import tempfile
import uuid
import os
from src.util import jobs
app = FastAPI()

current_output_file = None

app.mount("/static", StaticFiles(directory="./static"), name="static")
templates = Jinja2Templates(directory="./templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/image", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("image.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    # Read image in memory buffer
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    output_filename = predict_and_save(img)

    return templates.TemplateResponse("image.html", {"request": request, "result_image": "/" + output_filename})



@app.get("/video", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("video.html", {"request": request})


@app.post("/process-video")
async def process_video_route(file: UploadFile, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    with open(temp_file.name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_path = f"{output_dir_video}/{job_id}.mp4"
    
    jobs[job_id] = {
        "input_path": temp_file.name,
        "output_path": output_path,
        "progress": 0,
    }

    # Run background task with job_id
    background_tasks.add_task(process_video, temp_file.name, output_path, job_id)

    return {"message": "Processing started", "job_id": job_id}

@app.get("/progress/{job_id}")
async def progress_route(job_id: str):
    async def event_stream():
        prev = -1
        while True:
            prog = jobs.get(job_id, {}).get("progress", 0)
            if prog != prev:
                yield f"data: {prog}\n\n"
                prev = prog
            if prog >= 100:
                break
            await asyncio.sleep(0.5)
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/output-video/{job_id}")
async def output_video_route(job_id: str):
    output_path = jobs.get(job_id, {}).get("output_path")
    if output_path and os.path.exists(output_path):
        return FileResponse(output_path, media_type="video/mp4", filename=f"{job_id}.mp4")
    return {"error": "Output video not ready"}