import cv2
import numpy as np
import onnxruntime as ort

# ONNX modelini yuklash
session = ort.InferenceSession("/home/bahrombek/Desktop/Auto-proctoring/tester/yolo11n.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Kamerani ochish
cap = cv2.VideoCapture(0)

def preprocess(image, input_size=640):
    img = cv2.resize(image, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
    return img

def postprocess(outputs, conf_thres=0.5, iou_thres=0.5):
    predictions = outputs[0]  # Shape: [1, 5, 8400] yoki boshqa
    boxes, scores, class_ids = [], [], []

    for pred in predictions[0]:
        conf = pred[4]  # Confidence
        if conf > conf_thres:
            x, y, w, h = pred[0:4]
            boxes.append([x - w / 2, y - h / 2, w, h])
            scores.append(conf)
            class_ids.append(0)  # Class ID’ni o‘zgartirish kerak bo‘lsa, qo‘shing

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    return [(boxes[i], scores[i], class_ids[i]) for i in indices]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Rasmni tayyorlash
    input_img = preprocess(frame)

    # Inference
    outputs = session.run([output_name], {input_name: input_img})

    # Natijalarni qayta ishlash
    detections = postprocess(outputs)

    # Natijalarni chizish
    for (box, score, class_id) in detections:
        x, y, w, h = box
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"Class {class_id}: {score:.2f}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 ONNX Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()