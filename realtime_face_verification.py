import dlib
import cv2
import numpy as np
import lmdb
import pickle

# Modellarni yuklash
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Yuz embeddingini olish
def get_face_embedding(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(frame_rgb)
    if len(faces) == 0:
        return None
    shape = predictor(frame_rgb, faces[0])
    face_descriptor = facerec.compute_face_descriptor(frame_rgb, shape)
    return np.array(face_descriptor)

# LMDB dan embeddingni olish
def get_stored_embedding(lmdb_path, user_id):
    env = lmdb.open(lmdb_path, readonly=True)
    with env.begin() as txn:
        data = txn.get(user_id.encode('utf-8'))
        if data is None:
            return None
        return pickle.loads(data)

# Barcha embeddinglarni olish
def get_all_embeddings(lmdb_path):
    embeddings = {}
    env = lmdb.open(lmdb_path, readonly=True)
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            embeddings[key.decode()] = pickle.loads(val)
    return embeddings

# Embeddinglar o‘rtasidagi masofa
def compare_faces(embedding1, embedding2):
    if embedding1 is None or embedding2 is None:
        return False
    distance = np.linalg.norm(embedding1 - embedding2)
    return distance < 0.6

# Kimligini aniqlash
def identify_user(current_embedding, all_embeddings):
    for user_id, stored_embedding in all_embeddings.items():
        if compare_faces(current_embedding, stored_embedding):
            return user_id
    return None

# 1. Faqat bir user_id bilan tekshirish
def verify_single_user(lmdb_path, user_id):
    cap = cv2.VideoCapture(0)
    stored_embedding = get_stored_embedding(lmdb_path, user_id)
    if stored_embedding is None:
        print(f"{user_id} uchun embedding topilmadi.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        embedding = get_face_embedding(frame)
        text = "Yuz topilmadi"
        color = (255, 0, 0)

        if embedding is not None:
            if compare_faces(embedding, stored_embedding):
                text = f"{user_id} tasdiqlandi ✅"
                color = (0, 255, 0)
            else:
                text = f"{user_id} mos kelmadi ❌"
                color = (0, 0, 255)

        faces = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.putText(frame, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("1-Foydalanuvchini tekshirish", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 2. Barcha user_id lar bilan solishtirish
def identify_from_all_users(lmdb_path):
    cap = cv2.VideoCapture(0)
    all_embeddings = get_all_embeddings(lmdb_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        embedding = get_face_embedding(frame)
        text = "Yuz topilmadi"
        color = (255, 0, 0)

        if embedding is not None:
            user_id = identify_user(embedding, all_embeddings)
            if user_id:
                text = f"Topildi: {user_id} ✅"
                color = (0, 255, 0)
            else:
                text = "Mos keladigan foydalanuvchi topilmadi ❌"
                color = (0, 0, 255)

        faces = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.putText(frame, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("2-Barcha foydalanuvchilarni aniqlash", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- Interaktiv tanlov ---
if __name__ == "__main__":
    lmdb_path = "face_embeddings_lmdb"

    print("📌 Yuz tekshirish usulini tanlang:")
    print("1 - Faqat bitta user_id bilan tekshirish")
    print("2 - Barcha foydalanuvchilar orasidan aniqlash")

    choice = input("Tanlovingiz (1/2): ").strip()

    if choice == "1":
        env = lmdb.open(lmdb_path, readonly=True)
        with env.begin() as txn:
            cursor = txn.cursor()
            print("📋 Mavjud foydalanuvchilar:")
            for key, _ in cursor:
                print(" -", key.decode())
        user_id = input("🔑 Tekshiriladigan user_id ni kiriting: ").strip()
        verify_single_user(lmdb_path, user_id)
    elif choice == "2":
        identify_from_all_users(lmdb_path)
    else:
        print("❌ Noto‘g‘ri tanlov. 1 yoki 2 ni tanlang.")
