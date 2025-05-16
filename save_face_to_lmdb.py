import dlib
import cv2
import numpy as np
import lmdb
import pickle
import os
import uuid

# Modellarni yuklash
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Yuz embeddingini olish funksiyasi
def get_face_embedding(image_path):
    # Rasmni o'qish
    img = cv2.imread(image_path)
    if img is None:
        print(f"Rasm yuklanmadi: {image_path}")
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Yuzni aniqlash
    faces = detector(img_rgb)
    if len(faces) == 0:
        print("Yuz topilmadi")
        return None
    
    # Birinchi yuz uchun landmark yuzning landmarklarini aniqlash
    shape = predictor(img_rgb, faces[0])
    
    # Yuz embeddingini hisoblash
    face_descriptor = facerec.compute_face_descriptor(img_rgb, shape)
    return np.array(face_descriptor)

# LMDB ga yuz vektorini saqlash funksiyasi
def save_to_lmdb(image_path, lmdb_path, user_id=None):
    # Yuz embeddingini olish
    embedding = get_face_embedding(image_path)
    if embedding is None:
        print("Embedding hisoblanmadi")
        return
    
    # Agar user_id berilmagan bo'lsa, UUID generatsiya qilish
    if user_id is None:
        user_id = str(uuid.uuid4())
    
    # LMDB muhitini ochish
    env = lmdb.open(lmdb_path, map_size=10485760)  # 10MB
    with env.begin(write=True) as txn:
        # Embeddingni pickle orqali seriyalizatsiya qilish
        serialized_embedding = pickle.dumps(embedding)
        # user_id ni bayt sifatida saqlash
        txn.put(user_id.encode('utf-8'), serialized_embedding)
    
    print(f"Yuz vektori LMDB ga saqlandi: {user_id}")

# Test qilish
if __name__ == "__main__":
    image_path = "2025-05-15_15:25:11.jpg"  # Rasm fayli yo'li
    lmdb_path = "face_embeddings_lmdb"  # LMDB ma'lumotlar bazasi yo'li
    user_id = "user_002"  # Foydalanuvchi ID (ixtiyoriy)
    
    # Agar LMDB papkasi mavjud bo'lmasa, yaratish
    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path)
    
    save_to_lmdb(image_path, lmdb_path, user_id)