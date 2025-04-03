# src/face_database.py
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import lmdb
import os
import pickle

class FaceDatabase:
    def __init__(self, lmdb_dir="lmdb_data", providers=['CPUExecutionProvider']):
        self.app = FaceAnalysis(providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.lmdb_dir = lmdb_dir
        if not os.path.exists(lmdb_dir):
            os.makedirs(lmdb_dir)
    
    def load_image(self, file_path):
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def get_face_embedding(self, image):
        faces = self.app.get(image)
        if len(faces) == 0:
            return None, "Yuz topilmadi"
        embedding = faces[0].normed_embedding
        return embedding, faces[0].bbox
    
    def save_to_lmdb(self, image_path, user_id):
        image = self.load_image(image_path)
        embedding, bbox = self.get_face_embedding(image)
        if embedding is None:
            print(f"Xato: {image_path} faylida yuz topilmadi")
            return False
        
        env = lmdb.open(self.lmdb_dir, map_size=10485760)
        with env.begin(write=True) as txn:
            key = str(user_id).encode('utf-8')
            value = pickle.dumps(embedding)
            txn.put(key, value)
        
        print(f"ID: {user_id} uchun embedding LMDB ga saqlandi")
        env.close()
        return True
    
    def read_from_lmdb(self, user_id):
        env = lmdb.open(self.lmdb_dir, readonly=True)
        with env.begin() as txn:
            key = str(user_id).encode('utf-8')
            value = txn.get(key)
            if value is None:
                print(f"ID: {user_id} uchun ma'lumot topilmadi")
                env.close()
                return None
            embedding = pickle.loads(value)
        env.close()
        return embedding
    
    def load_all_embeddings(self):
        env = lmdb.open(self.lmdb_dir, readonly=True)
        embeddings_db = {}
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                user_id = key.decode('utf-8')
                embedding = pickle.loads(value)
                embeddings_db[user_id] = embedding
        env.close()
        return embeddings_db

if __name__ == "__main__":
    image_path = "/home/bahrombek/Desktop/Avtoproktoring/src/saved_images/captured_face_1743682625.jpg"
    user_id = "Bahrombek"
    face_db = FaceDatabase()
    face_db.save_to_lmdb(image_path, user_id)
    embedding = face_db.read_from_lmdb(user_id)
    if embedding is not None:
        print(f"ID: {user_id} uchun embedding uzunligi: {len(embedding)}")
        print(f"Embedding namunasi: {embedding[:5]}")