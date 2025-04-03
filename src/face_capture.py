# src/face_capture.py
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import time
import os

class FaceCapture:
    def __init__(self, providers=['CPUExecutionProvider']):
        self.app = FaceAnalysis(providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
    
    def get_face_embedding(self, image):
        faces = self.app.get(image)
        if len(faces) == 0:
            return None, "Yuz topilmadi"
        return faces[0].normed_embedding, faces[0].bbox
    
    def save_image(self, frame, save_dir="saved_images", filename="captured_face.jpg"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"Rasm {save_path} ga saqlandi.")
        return save_path
    
    def capture_face_with_delay(self, save_dir="saved_images"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Kamera ochilmadi!")
            return None
        
        face_detected = False
        detection_time = None
        saved_path = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Kadr olishda xato!")
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            embedding, bbox = self.get_face_embedding(rgb_frame)
            
            if embedding is not None:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Yuz topildi", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                if not face_detected:
                    face_detected = True
                    detection_time = time.time()
                    print("Yuz aniqlandi! 5 sekund kutilmoqda...")

                if face_detected and (time.time() - detection_time >= 5):
                    saved_path = self.save_image(rgb_frame, save_dir, f"captured_face_{int(time.time())}.jpg")
                    break
            else:
                cv2.putText(frame, "Yuz topilmadi", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                face_detected = False
                detection_time = None

            cv2.imshow("Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Dastur foydalanuvchi tomonidan to'xtatildi.")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return saved_path

if __name__ == "__main__":
    save_dir = "saved_images"
    face_capture = FaceCapture()
    face_capture.capture_face_with_delay(save_dir)