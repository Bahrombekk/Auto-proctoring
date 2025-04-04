# config.py
import os

# Umumiy sozlamalar
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LMDB_DIR = os.path.join(BASE_DIR, "/home/bahrombek/Desktop/Auto-proctoring/src/lmdb_data")
SAVE_DIR = os.path.join(BASE_DIR, "saved_images")
VIDEO_DIR = os.path.join(BASE_DIR, "recorded_videos")

# YOLO modellari uchun yo'llar
PERSON_MODEL_PATH = "yolov8n.pt"  # Odam aniqlash uchun YOLOv8 modeli
PHONE_MODEL_PATH = os.path.join(BASE_DIR, "manitoring", "best.pt")  # Telefon aniqlash uchun maxsus model

# Thresholdlar
FACE_SIMILARITY_THRESHOLD = 0.4  # Yuz o'xshashligi uchun chegaraviy qiymat
DETECTION_CONF_THRESHOLD = 0.5  # Ob'ekt aniqlash uchun ishonchlilik chegarasi

# Video yozish sozlamalari
VIDEO_DURATION = 10  # Sekundlarda oldingi va keyingi video uzunligi
FPS = 30  # Kadrlar soni sekundda
FRAME_BUFFER_SIZE = FPS * VIDEO_DURATION * 2  # 20 sekundlik bufer (10s oldin + 10s keyin)

# Kamera sozlamalari
CAMERA_INDEX = 0  # Standart kamera indeksi

# Direktoriyalarni yaratish
for directory in [LMDB_DIR, SAVE_DIR, VIDEO_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)