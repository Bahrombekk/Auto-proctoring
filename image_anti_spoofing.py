import os
import cv2
import numpy as np
import warnings
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings('ignore')

# Yuzni aniqlash uchun Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Yuzni aniqlash va anti-spoofing tahlili (tasvir uchun)
def anti_spoofing_image(image_path):
    model_dir = "./resources/anti_spoof_models"  # Model kutubxonasi
    device_id = 0  # GPU ID

    # Tasvirni o'qish
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError("Tasvirni o'qib bo'lmadi. Fayl yo'lini tekshiring.")

    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    
    # Rangli tasvirni kulrangga aylantirish
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Yuzlarni aniqlash
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    is_real_face = False  # Soxta yoki haqiqiy yuz aniqlanganini kuzatib borish uchun flag

    for (x, y, w, h) in faces:
        # Yuzni kesish
        face = frame[y:y+h, x:x+w]
        # Yuzni o'lchash
        face_resized = cv2.resize(face, (60, 80), interpolation=cv2.INTER_LINEAR)
        
        # Anti-spoofing tahlili
        image_bbox = (x, y, w, h)  # Yuzning koordinatalari
        prediction = np.zeros((1, 3))
        
        # Modelni ishlatish
        for model_name in os.listdir(model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": face_resized,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            prediction += model_test.predict(img, os.path.join(model_dir, model_name))

        # Natijani aniqlash
        label = np.argmax(prediction)
        value = prediction[0][label] / 2
        
        # Threshold qiymati
        threshold = 0.1  # O'zgartirilgan threshold
        if value < threshold:
            label = 0  # Soxta yuz deb belgilash
        
        if label == 1:
            result_text = "Haqiqiy Yuz: {:.2f}".format(value)
            color = (255, 0, 0)  # Qizil
            is_real_face = True  # Haqiqiy yuz topildi
        else:
            result_text = "Soxta Yuz: {:.2f}".format(value)
            color = (0, 0, 255)  # Ko'k
        
        # Yuzga to'rtburchak va matn qo'shish
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, result_text, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, color)

    # Natijani saqlash yoki ko'rsatish
    output_path = "output_image.jpg"
    cv2.imwrite(output_path, frame)  # Natijani saqlash
    # cv2.imshow("Result", frame)  # Agar ko'rsatish kerak bo'lsa
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return frame, is_real_face  # Yangilangan tasvir va haqiqiylik flagi qaytariladi

# Kodni ishlatish
image_path = "path_to_your_image.jpg"  # Tasvir faylining yo'lini kiriting
processed_image, is_real = anti_spoofing_image(image_path)
print("Haqiqiy yuz aniqlandi" if is_real else "Soxta yuz aniqlandi")