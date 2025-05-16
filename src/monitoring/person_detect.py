# src/monitoring/person_detect.py
import cv2
from ultralytics import YOLO
from config import DETECTION_CONF_THRESHOLD, ENABLE_PERSON_DETECTION
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def detect_persons(frame, model):
    if not ENABLE_PERSON_DETECTION:
        logging.debug("Person detection is disabled.")
        return 0, frame

    results = model(frame)
    person_count = 0
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 0 and box.conf[0] > DETECTION_CONF_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                person_count += 1
    return person_count, frame