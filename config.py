import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory paths
LMDB_DIR = os.path.join(BASE_DIR, "lmdb_data")
SAVE_DIR = os.path.join(BASE_DIR, "saved_images")
VIDEO_DIR = os.path.join(BASE_DIR, "recorded_videos")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# YOLO model paths
PERSON_MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n.onnx")  # Person detection model
PHONE_MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")        # Phone detection model

# Feature toggle switches
ENABLE_PERSON_DETECTION = False   # Enable/disable person detection
ENABLE_PHONE_DETECTION = False    # Enable/disable phone detection
ENABLE_FACE_RECOGNITION = False   # Enable/disable face recognition
ENABLE_PROCTORING = True         # Enable/disable proctoring rules

# Thresholds
FACE_SIMILARITY_THRESHOLD = 0.4  # Face similarity threshold
DETECTION_CONF_THRESHOLD = 0.5   # Object detection confidence threshold

# Video recording settings
VIDEO_DURATION = 10  # Seconds for before and after violation video
FPS = 30             # Frames per second
FRAME_BUFFER_SIZE = FPS * VIDEO_DURATION * 2  # 20-second buffer (10s before + 10s after)

# Camera settings
CAMERA_INDEX = 0  # Default camera index

# Create directories if they don't exist
for directory in [LMDB_DIR, SAVE_DIR, VIDEO_DIR, MODEL_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)