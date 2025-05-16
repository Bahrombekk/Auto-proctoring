# src/face_database.py
import cv2
import numpy as np
import lmdb
import os
import pickle
import logging
from config import LMDB_DIR, LMDB_DIR_DLIB, LMDB_DIR_INSIGHTFACE

# Try to import InsightFace and Dlib if available
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available. Install with: pip install insightface==0.7.3")

try:
    import dlib
    import uuid
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    logging.warning("Dlib not available. Install with: pip install dlib")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FaceDatabase:
    def __init__(self, lmdb_dir=LMDB_DIR, model_type=None):
        """
        Initialize the face database with specific model type
        
        Args:
            lmdb_dir: Directory for LMDB database
            model_type: 'insightface' or 'dlib'
        """
        self.lmdb_dir = lmdb_dir
        self.model_type = model_type or "insightface"  # Default to InsightFace if not specified
        
        # Create directory if it doesn't exist
        if not os.path.exists(self.lmdb_dir):
            try:
                os.makedirs(self.lmdb_dir)
                logging.info(f"Created directory: {self.lmdb_dir}")
            except Exception as e:
                logging.error(f"Failed to create directory {self.lmdb_dir}: {str(e)}")
        
        # Initialize appropriate face recognition model
        if self.model_type == "insightface":
            if INSIGHTFACE_AVAILABLE:
                try:
                    self.face_app = FaceAnalysis(allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
                    self.face_app.prepare(ctx_id=0, det_size=(640, 640))
                    logging.info("InsightFace model initialized successfully.")
                except Exception as e:
                    logging.error(f"Failed to initialize InsightFace: {str(e)}")
                    self.face_app = None
            else:
                logging.error("InsightFace library not available.")
                self.face_app = None
                
        elif self.model_type == "dlib":
            if DLIB_AVAILABLE:
                try:
                    # Initialize Dlib's face detector and shape predictor
                    self.face_detector = dlib.get_frontal_face_detector()
                    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                              "models", "shape_predictor_68_face_landmarks.dat")
                    
                    if os.path.exists(model_path):
                        self.shape_predictor = dlib.shape_predictor(model_path)
                    else:
                        logging.error(f"Dlib shape predictor model not found at {model_path}")
                        self.shape_predictor = None
                        
                    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                              "models", "dlib_face_recognition_resnet_model_v1.dat")
                    
                    if os.path.exists(model_path):
                        self.face_recognizer = dlib.face_recognition_model_v1(model_path)
                        logging.info("Dlib model initialized successfully.")
                    else:
                        logging.error(f"Dlib face recognition model not found at {model_path}")
                        self.face_recognizer = None
                except Exception as e:
                    logging.error(f"Failed to initialize Dlib: {str(e)}")
                    self.face_detector = None
                    self.shape_predictor = None
                    self.face_recognizer = None
            else:
                logging.error("Dlib library not available.")
                self.face_detector = None
                self.shape_predictor = None
                self.face_recognizer = None
        else:
            logging.error(f"Unsupported model type: {self.model_type}")

    def get_face_embedding(self, image):
        """
        Extract face embedding from an image
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            embedding: Face embedding vector
            bbox: Bounding box coordinates [x1, y1, x2, y2]
        """
        if self.model_type == "insightface" and self.face_app is not None:
            try:
                faces = self.face_app.get(image)
                if len(faces) == 0:
                    logging.warning("No face detected with InsightFace")
                    return None, None
                    
                # Get the largest face by area
                largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                embedding = largest_face.embedding
                bbox = largest_face.bbox
                return embedding, bbox
            except Exception as e:
                logging.error(f"InsightFace embedding extraction error: {str(e)}")
                return None, None
                
        elif self.model_type == "dlib" and self.face_detector is not None and self.shape_predictor is not None and self.face_recognizer is not None:
            try:
                # Convert RGB to BGR for Dlib
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if len(image.shape) == 3 and image.shape[2] == 3 else image
                
                # Detect faces
                faces = self.face_detector(image_bgr)
                if len(faces) == 0:
                    logging.warning("No face detected with Dlib")
                    return None, None
                
                # Get the largest face by area
                largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
                
                # Get face landmarks
                shape = self.shape_predictor(image_bgr, largest_face)
                
                # Compute face embedding
                face_embedding = self.face_recognizer.compute_face_descriptor(image_bgr, shape)
                face_embedding = np.array(face_embedding)
                
                # Convert dlib rectangle to bbox [x1, y1, x2, y2]
                bbox = [largest_face.left(), largest_face.top(), 
                        largest_face.right(), largest_face.bottom()]
                
                return face_embedding, bbox
            except Exception as e:
                logging.error(f"Dlib embedding extraction error: {str(e)}")
                return None, None
        else:
            logging.error(f"Face embedding extraction not available for {self.model_type}")
            return None, None

    def save_to_lmdb(self, image_path, user_id):
        """
        Save face embedding to LMDB database
        
        Args:
            image_path: Path to the image file
            user_id: User ID string
            
        Returns:
            bool: Success or failure
        """
        try:
            # Check if image exists
            if not os.path.exists(image_path):
                logging.error(f"Image file not found: {image_path}")
                return False
                
            # Read and process image
            image = cv2.imread(image_path)
            if image is None:
                logging.error(f"Failed to read image: {image_path}")
                return False
                
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get face embedding
            embedding, bbox = self.get_face_embedding(image_rgb)
            if embedding is None:
                logging.error(f"No face detected in image: {image_path}")
                return False
                
            # Save embedding to LMDB
            env = lmdb.open(self.lmdb_dir, map_size=10485760)
            with env.begin(write=True) as txn:
                key = str(user_id).encode('utf-8')
                value = pickle.dumps(embedding)
                txn.put(key, value)
            env.close()
            
            logging.info(f"User {user_id} saved to {self.model_type} database successfully.")
            return True
        except Exception as e:
            logging.error(f"Error saving face to database: {str(e)}")
            return False

    def compare_face(self, embedding1, embedding2):
        """
        Compare two face embeddings
        
        Args:
            embedding1, embedding2: Face embedding vectors
            
        Returns:
            float: Similarity score or distance
        """
        if self.model_type == "insightface":
            # InsightFace uses cosine similarity (higher is better match)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 == 0 or norm2 == 0:
                return 0
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return similarity
        elif self.model_type == "dlib":
            # Dlib uses Euclidean distance (lower is better match)
            distance = np.linalg.norm(embedding1 - embedding2)
            return distance
        else:
            logging.error(f"Face comparison not available for {self.model_type}")
            return -1

    def compare_face_with_all(self, test_embedding):
        """
        Compare test embedding with all faces in the database
        
        Args:
            test_embedding: Face embedding vector to test
            
        Returns:
            tuple: (best_match_id, similarity_score)
        """
        try:
            if not os.path.exists(self.lmdb_dir):
                logging.error(f"Database directory not found: {self.lmdb_dir}")
                return None, -1
                
            if test_embedding is None:
                logging.error("Test embedding is None")
                return None, -1
                
            env = lmdb.open(self.lmdb_dir, readonly=True)
            best_match_id = None
            
            if self.model_type == "insightface":
                best_similarity = -1  # Max similarity for InsightFace
            else:  # dlib
                best_similarity = float('inf')  # Min distance for Dlib
                
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    user_id = key.decode('utf-8')
                    db_embedding = pickle.loads(value)
                    
                    score = self.compare_face(test_embedding, db_embedding)
                    
                    if self.model_type == "insightface":
                        # For InsightFace, higher similarity is better
                        if score > best_similarity:
                            best_similarity = score
                            best_match_id = user_id
                    else:  # dlib
                        # For Dlib, lower distance is better
                        if score < best_similarity:
                            best_similarity = score
                            best_match_id = user_id
            
            env.close()
            return best_match_id, best_similarity
        except Exception as e:
            logging.error(f"Error comparing face with database: {str(e)}")
            return None, -1