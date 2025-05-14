import cv2
import numpy as np
from insightface.app import FaceAnalysis
import lmdb
import pickle
from .face_database import FaceDatabase
from config import LMDB_DIR, FACE_SIMILARITY_THRESHOLD, ENABLE_FACE_RECOGNITION
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FaceRecognition:
    def __init__(self, lmdb_dir=LMDB_DIR, providers=['CPUExecutionProvider']):
        self.app = FaceAnalysis(providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.lmdb_dir = lmdb_dir
        self.face_db = FaceDatabase(lmdb_dir, providers) if ENABLE_FACE_RECOGNITION else None
    
    def get_face_embedding(self, image):
        if not ENABLE_FACE_RECOGNITION:
            logging.debug("Face recognition is disabled.")
            return None, None
        faces = self.app.get(image)
        if len(faces) == 0:
            return None, None
        embedding = faces[0].normed_embedding
        return embedding, faces[0].bbox
    
    def compare_face_with_target(self, embedding, target_id, threshold=FACE_SIMILARITY_THRESHOLD):
        if not ENABLE_FACE_RECOGNITION or embedding is None or self.face_db is None:
            return None, -1
        db_embedding = self.face_db.read_from_lmdb(target_id)
        if db_embedding is None:
            return None, -1
        similarity = np.dot(embedding, db_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(db_embedding))
        if similarity > threshold:
            return target_id, similarity
        return None, similarity
    
    def compare_face_with_all(self, embedding, threshold=FACE_SIMILARITY_THRESHOLD):
        if not ENABLE_FACE_RECOGNITION or embedding is None or self.face_db is None:
            return None, -1
        db_embeddings = self.face_db.load_all_embeddings()
        if not db_embeddings:
            return None, -1
        best_match_id = None
        best_similarity = -1
        for user_id, db_embedding in db_embeddings.items():
            similarity = np.dot(embedding, db_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(db_embedding))
            if similarity > threshold and similarity > best_similarity:
                best_match_id = user_id
                best_similarity = similarity
        return best_match_id, best_similarity

if __name__ == "__main__":
    face_recognizer = FaceRecognition()