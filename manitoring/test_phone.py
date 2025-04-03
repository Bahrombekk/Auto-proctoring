import cv2
from ultralytics import YOLO
import numpy as np

# Modelni yuklash
model = YOLO("/home/bahrombek/Desktop/Avtoproktoring/manitoring/best.pt")

img=cv2.imread("/home/bahrombek/Desktop/Avtoproktoring/manitoring/temp.jpg")
frame=img
results = model("/home/bahrombek/Desktop/Avtoproktoring/manitoring/temp.jpg")

# Natijalarni ekranga chiqarish
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # To'rtburchak koordinatalari
        conf = box.conf[0]  # Ishonchlilik darajasi
        cls = int(box.cls[0])  # Klass indeksi
        if conf >0.7:
            if cls == 0:  # 0 klass - odam
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"phone {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("YOLOv8 Person Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()