# config.py
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory paths
# Default LMDB directory
LMDB_DIR = os.path.join(BASE_DIR, "lmdb_data")  # Generic LMDB database directory
LMDB_DIR_DLIB = os.path.join(BASE_DIR, "lmdb_data_dlib")  # Dlib ma'lumotlar bazasi papkasi
LMDB_DIR_INSIGHTFACE = os.path.join(BASE_DIR, "lmdb_data_insightface")  # InsightFace ma'lumotlar bazasi
SAVE_DIR = os.path.join(BASE_DIR, "saved_images")  # Saqlangan rasmlar papkasi
VIDEO_DIR = os.path.join(BASE_DIR, "recorded_videos")  # Yozilgan videolar papkasi
MODEL_DIR = os.path.join(BASE_DIR, "models")  # Modellar papkasi

# YOLO model paths
PERSON_MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n.onnx")  # Odam aniqlash modeli
PHONE_MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")  # Telefon aniqlash modeli

# Dlib face detection model path
DLIB_MODEL_PATH = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")  # Dlib yuz landmark modeli

# Face recognition model path
FACE_RECOGNITION_MODEL_PATH = os.path.join(MODEL_DIR, "dlib_face_recognition_resnet_model_v1.dat")  # Dlib yuz tanish modeli

# Feature toggle switches
ENABLE_PERSON_DETECTION = True   # Odam aniqlashni yoqish/o'chirish
ENABLE_PHONE_DETECTION = True    # Telefon aniqlashni yoqish/o'chirish
ENABLE_FACE_RECOGNITION = True   # Yuz tanishni yoqish/o'chirish (umumiy boshqaruv)
ENABLE_PROCTORING = True         # Proctoring qoidalarini yoqish/o'chirish
ENABLE_DLIB = True               # Dlib yuz aniqlashni yoqish/o'chirish (real vaqtda)
ENABLE_INSIGHTFACE = False        # InsightFace yuz aniqlashni yoqish/o'chirish (real vaqtda)
ENABLE_DLIB_LMDB_ADD_FACE = True # Dlib embeddinglarini LMDB ga qo'shishni yoqish/o'chirish
ENABLE_INSIGHTFACE_LMDB_ADD_FACE = True  # InsightFace embeddinglarini LMDB ga qo'shishni yoqish/o'chirish

# Thresholds
FACE_SIMILARITY_THRESHOLD = 0.4  # InsightFace uchun kosinus o'xshashlik chegarasi (0-1 oralig'i)
DLIB_DISTANCE_THRESHOLD = 0.6    # Dlib uchun Evklid masofasi chegarasi (0.6 dan kichik bo'lsa mos)
DETECTION_CONF_THRESHOLD = 0.5   # Ob'ekt aniqlash ishonch chegarasi (YOLO uchun)

# Video recording settings
VIDEO_DURATION = 10  # Qoidabuzarlikdan oldin va keyin video uchun soniyalar
FPS = 30             # Kadrlar soni sekundiga
FRAME_BUFFER_SIZE = FPS * VIDEO_DURATION * 2  # 20 soniyalik bufer (10s oldin + 10s keyin)

# Camera settings
CAMERA_INDEX = 0  # Standart kamera indeksi

# Create directories if they don't exist
for directory in [LMDB_DIR_DLIB, LMDB_DIR_INSIGHTFACE, SAVE_DIR, VIDEO_DIR, MODEL_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Model fayllarining mavjudligini tekshirish
for model_path in [PERSON_MODEL_PATH, PHONE_MODEL_PATH, DLIB_MODEL_PATH, FACE_RECOGNITION_MODEL_PATH]:
    if not os.path.exists(model_path):
        logging.error(f"Model fayli topilmadi: {model_path}")

# Sozlamalarni validatsiya qilish
def validate_config():
    if ENABLE_DLIB_LMDB_ADD_FACE and not ENABLE_DLIB:
        logging.warning("ENABLE_DLIB_LMDB_ADD_FACE True, lekin ENABLE_DLIB False. Dlib embeddinglarini real vaqtda taqqoslash ishlamaydi.")
    if ENABLE_INSIGHTFACE_LMDB_ADD_FACE and not ENABLE_INSIGHTFACE:
        logging.warning("ENABLE_INSIGHTFACE_LMDB_ADD_FACE True, lekin ENABLE_INSIGHTFACE False. InsightFace embeddinglarini real vaqtda taqqoslash ishlamaydi.")
    if ENABLE_FACE_RECOGNITION and not (ENABLE_DLIB or ENABLE_INSIGHTFACE):
        logging.warning("ENABLE_FACE_RECOGNITION True, lekin hech qanday yuz aniqlash (Dlib yoki InsightFace) yoqilmagan.")

validate_config()