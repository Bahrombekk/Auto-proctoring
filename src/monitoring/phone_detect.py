import cv2
from ultralytics import YOLO
from config import DETECTION_CONF_THRESHOLD, ENABLE_PHONE_DETECTION
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def detect_phone(frame, model):
    if not ENABLE_PHONE_DETECTION:
        logging.debug("Phone detection is disabled.")
        return False, frame

    results = model(frame)
    phone_detected = False
    for result in results:
        for box in result.boxes:
            if box.conf[0] > DETECTION_CONF_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Phone {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                phone_detected = True
    return phone_detected, frame