# src/face_recognition.py
import logging
import cv2
import numpy as np
import lmdb
import pickle
from src.face_database import FaceDatabase

class FaceRecognition:
    def __init__(self, lmdb_dir, model_type=None):
        """
        Initialize face recognition system
        
        Args:
            lmdb_dir: Directory for LMDB database
            model_type: 'insightface' or 'dlib'
        """
        self.model_type = model_type or "insightface"  # Default to InsightFace if not specified
        self.lmdb_dir = lmdb_dir
        
        # Initialize face database with appropriate model
        try:
            self.face_db = FaceDatabase(lmdb_dir, model_type)
            logging.info(f"Face recognition initialized with {model_type} model")
        except Exception as e:
            logging.error(f"Failed to initialize face database: {str(e)}")
            self.face_db = None
    
    def get_face_embedding(self, image, model_type=None):
        """
        Extract face embedding from an image
        
        Args:
            image: RGB image as numpy array
            model_type: Optional override for model type
            
        Returns:
            embedding: Face embedding vector
            bbox: Bounding box coordinates [x1, y1, x2, y2]
        """
        # Use instance model_type if not specified
        model_type = model_type or self.model_type
        
        if self.face_db is None:
            logging.error("Face database not initialized")
            return None, None
            
        try:
            # Use face database to get embedding
            embedding, bbox = self.face_db.get_face_embedding(image)
            return embedding, bbox
        except Exception as e:
            logging.error(f"Error getting face embedding: {str(e)}")
            return None, None
    
    def compare_face_with_all(self, test_embedding, model_type=None):
        """
        Compare test embedding with all faces in the database
        
        Args:
            test_embedding: Face embedding vector to test
            model_type: Optional override for model type
            
        Returns:
            tuple: (best_match_id, similarity_score or distance)
        """
        # Use instance model_type if not specified
        model_type = model_type or self.model_type
        
        if self.face_db is None:
            logging.error("Face database not initialized")
            return None, -1
            
        try:
            # Use face database to compare with all stored faces
            return self.face_db.compare_face_with_all(test_embedding)
        except Exception as e:
            logging.error(f"Error comparing face with database: {str(e)}")
            return None, -1