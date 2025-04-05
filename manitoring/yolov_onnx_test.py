from ultralytics import YOLO
import cv2
import time
import numpy as np
import onnxruntime as ort

# 1. PyTorch modelini yuklash
pt_model = YOLO("/home/bahrombek/Desktop/Auto-proctoring/manitoring/yolo11n.pt")  # .pt fayl yo‘li

# 2. ONNX modelini yuklash
onnx_model_path = "/home/bahrombek/Desktop/Auto-proctoring/manitoring/yolo11n.onnx"  # .onnx fayl yo‘li
onnx_session = ort.InferenceSession(onnx_model_path)
onnx_input_name = onnx_session.get_inputs()[0].name
onnx_output_name = onnx_session.get_outputs()[0].name

# Kamerani ochish
cap = cv2.VideoCapture(0)

# ONNX uchun preprocessing funksiyasi
def preprocess_onnx(image, input_size=640):
    img = cv2.resize(image, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
    return img

# ONNX uchun postprocessing funksiyasi (soddalashtirilgan)
def postprocess_onnx(outputs, conf_thres=0.5):
    predictions = outputs[0]  # Shape: [1, num_boxes, num_classes + 5]
    boxes, scores, class_ids = [], [], []

    for pred in predictions[0]:
        conf = pred[4]
        if conf > conf_thres and int(pred[5:].argmax()) == 0:  # Faqat "person" (class 0)
            x, y, w, h = pred[:4]
            boxes.append([int(x - w / 2), int(y - h / 2), int(w), int(h)])
            scores.append(conf)
            class_ids.append(0)

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, 0.5)
    return [(boxes[i], scores[i]) for i in indices]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. PyTorch model bilan inference
    start_time_pt = time.time()
    pt_results = pt_model(frame)
    pt_time = time.time() - start_time_pt

    # PyTorch natijalarini chizish
    for result in pt_results:
        for box in result.boxes:
            if int(box.cls[0]) == 0:  # Faqat "person" class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"PT: Person {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 2. ONNX model bilan inference
    start_time_onnx = time.time()
    input_img = preprocess_onnx(frame)
    onnx_outputs = onnx_session.run([onnx_output_name], {onnx_input_name: input_img})
    onnx_detections = postprocess_onnx(onnx_outputs)
    onnx_time = time.time() - start_time_onnx

    # ONNX natijalarini chizish
    for (box, score) in onnx_detections:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, f"ONNX: Person {score:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Tezlikni ekranda ko‘rsatish
    cv2.putText(frame, f"PyTorch Time: {pt_time:.3f}s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"ONNX Time: {onnx_time:.3f}s", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("YOLOv8 PyTorch vs ONNX Speed Comparison", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()