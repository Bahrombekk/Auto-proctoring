# src/face_verification.py
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from config import FACE_SIMILARITY_THRESHOLD
import logging
import os
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FaceVerification:
    def __init__(self, providers=['CPUExecutionProvider']):
        self.app = FaceAnalysis(providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
    
    def load_image(self, file_path):
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def get_face_embedding(self, image):
        faces = self.app.get(image)
        if len(faces) == 0:
            return None, "No face detected"
        embedding = faces[0].normed_embedding
        return embedding, faces[0].bbox
    
    def compare_faces(self, embedding1, embedding2):
        if embedding1 is None or embedding2 is None:
            return False, "Face embeddings not found for comparison"
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        match = similarity > FACE_SIMILARITY_THRESHOLD
        return match, similarity
    
    def verify_faces(self, image_path1, image_path2):
        image1 = self.load_image(image_path1)
        image2 = self.load_image(image_path2)
        embedding1, _ = self.get_face_embedding(image1)
        embedding2, _ = self.get_face_embedding(image2)
        return self.compare_faces(embedding1, embedding2)

if __name__ == "__main__":
    from config import SAVE_DIR
    camera_image_path = os.path.join(SAVE_DIR, "captured_face_1743682625.jpg")
    passport_image_path = "/home/bahrombek/Desktop/Avtoproktoring/tester/Pasted image (2).png"
    verifier = FaceVerification()
    match, similarity = verifier.verify_faces(camera_image_path, passport_image_path)
    if match:
        logging.info(f"Images belong to the same person. Similarity: {similarity:.4f}")
    else:
        logging.info(f"Images belong to different persons. Similarity: {similarity:.4f}")