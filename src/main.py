from fastapi import FastAPI, Request, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
from src.model import predict_and_save, process_video, get_progress, output_dir
import asyncio
import shutil
import tempfile

app = FastAPI()

current_output_file = None

app.mount("/static", StaticFiles(directory="./static"), name="static")
templates = Jinja2Templates(directory="./templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    # Read image in memory buffer
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    output_filename = predict_and_save(img)

    return templates.TemplateResponse("index.html", {"request": request, "result_image": "/" + output_filename})


@app.get("/video", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("video.html", {"request": request})


@app.post("/process-video")
async def process_video_route(file: UploadFile, background_tasks: BackgroundTasks):
    global current_output_file

    # Save to temp file
    # Read video data into memory buffer
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    with open(temp_file.name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Generate unique output file name
    current_output_file = "output.mp4"

    # Start processing in background
    background_tasks.add_task(process_video, temp_file.name, current_output_file)

    return {"message": "Processing started"}

@app.get("/progress")
async def progress_route():
    async def event_stream():
        prev = -1
        while True:
            prog = get_progress()
            if prog != prev:
                yield f"data: {prog}\n\n"
                prev = prog
            if prog >= 100:
                break
            await asyncio.sleep(0.5)
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/output-video")
async def output_video_route():
    from os.path import join, exists
    output_path = join(output_dir, current_output_file)
    if exists(output_path):
        return FileResponse(output_path, media_type="video/x-msvideo", filename=current_output_file)
    return {"error": "Output video not ready"}