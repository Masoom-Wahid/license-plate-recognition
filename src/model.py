from ultralytics import YOLO
import uuid
import cv2
import os

model = YOLO("src/model/best.pt")

progress = 0
output_dir = "./static/videos"
os.makedirs(output_dir, exist_ok=True)


def predict_and_save(img):
    results = model.predict(source=img, conf=0.5, save=False)

    result_img = results[0].plot()

    output_filename = f"static/outputs/{uuid.uuid4()}.jpg"
    cv2.imwrite(output_filename, result_img)

    return output_filename


def process_video(input_path: str, output_name: str = "output.mp4"):
    global progress

    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    output_path = os.path.join(output_dir, output_name)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        frame_idx += 1
        progress = int((frame_idx / total_frames) * 100)

    cap.release()
    out.release()
    progress = 100

    return output_path

def get_progress():
    global progress
    return progress