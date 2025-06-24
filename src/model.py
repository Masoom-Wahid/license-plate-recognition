from ultralytics import YOLO
import uuid
import cv2


model = YOLO("/home/ahmadsohailraoufi/Desktop/dev/python/fastapi/license-plate/src/model/best.pt")


def predict_and_save(img):
    results = model.predict(source=img, conf=0.5, save=False)

    result_img = results[0].plot()

    output_filename = f"static/outputs/{uuid.uuid4()}.jpg"
    cv2.imwrite(output_filename, result_img)

    return output_filename