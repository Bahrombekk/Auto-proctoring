# src/face_capture.py
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import time
import os
from config import SAVE_DIR
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FaceCapture:
    def __init__(self, providers=['CPUExecutionProvider']):
        self.app = FaceAnalysis(providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
    
    def get_face_embedding(self, image):
        faces = self.app.get(image)
        if len(faces) == 0:
            return None, "No face detected"
        return faces[0].normed_embedding, faces[0].bbox
    
    def save_image(self, frame, save_dir=SAVE_DIR, filename="captured_face.jpg"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        logging.info(f"Image saved to {save_path}.")
        return save_path
    
    def capture_face_with_delay(self, save_dir=SAVE_DIR):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Camera could not be opened!")
            return None
        
        face_detected = False
        detection_time = None
        saved_path = None

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Error capturing frame!")
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            embedding, bbox = self.get_face_embedding(rgb_frame)
            
            if embedding is not None:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Face detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                if not face_detected:
                    face_detected = True
                    detection_time = time.time()
                    logging.info("Face detected! Waiting 5 seconds...")

                if face_detected and (time.time() - detection_time >= 5):
                    saved_path = self.save_image(rgb_frame, save_dir, f"captured_face_{int(time.time())}.jpg")
                    break
            else:
                cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                face_detected = False
                detection_time = None

            cv2.imshow("Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Program stopped by user.")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return saved_path

if __name__ == "__main__":
    face_capture = FaceCapture()
    face_capture.capture_face_with_delay()