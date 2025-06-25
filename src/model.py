from ultralytics import YOLO
import uuid
import cv2
import os
from src.util import jobs

model = YOLO("src/model/best.pt")

progress = 0
output_dir = "./static/outputs"
os.makedirs(output_dir, exist_ok=True)

output_dir_video = f"{output_dir}/videos"
output_dir_images = f"{output_dir}/images"
os.makedirs(output_dir_video, exist_ok=True)
os.makedirs(output_dir_images, exist_ok=True)


def predict_and_save(img):
    results = model.predict(source=img, conf=0.5, save=False)

    result_img = results[0].plot()

    output_filename = f"{output_dir_images}/{uuid.uuid4()}.jpg"
    cv2.imwrite(output_filename, result_img)

    return output_filename

def process_video(input_path: str, output_path: str, job_id: str):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error opening video file")
        jobs[job_id]["progress"] = 100
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 🧠 Process frame using YOLO model
        results = model.predict(source=frame, conf=0.5, save=False)
        processed_frame = results[0].plot()  # Draw boxes on frame

        out.write(processed_frame)

        count += 1
        jobs[job_id]["progress"] = int((count / total) * 100)

    cap.release()
    out.release()
    jobs[job_id]["progress"] = 100

def get_progress():
    global progress
    return progress