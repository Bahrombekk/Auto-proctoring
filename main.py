import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from src.face_recognition import FaceRecognition
from src.db_management import DBManager
from src.monitoring.person_detect import detect_persons
from src.monitoring.phone_detect import detect_phone
from src.monitoring.proctor import run_proctoring
from config import *
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load models if enabled
try:
    person_model = YOLO(PERSON_MODEL_PATH) if ENABLE_PERSON_DETECTION else None
    phone_model = YOLO(PHONE_MODEL_PATH) if ENABLE_PHONE_DETECTION else None
except Exception as e:
    logging.error(f"Failed to load YOLO models: {str(e)}")
    person_model = None
    phone_model = None

def real_time_monitoring():
    # Initialize face recognition systems separately if enabled
    insightface_recognizer = None
    dlib_recognizer = None
    
    if ENABLE_FACE_RECOGNITION:
        try:
            if ENABLE_INSIGHTFACE:
                insightface_recognizer = FaceRecognition(LMDB_DIR_INSIGHTFACE, model_type="insightface")
                logging.info("InsightFace recognition initialized.")
            
            if ENABLE_DLIB:
                dlib_recognizer = FaceRecognition(LMDB_DIR_DLIB, model_type="dlib")
                logging.info("Dlib recognition initialized.")
                
            if not insightface_recognizer and not dlib_recognizer:
                logging.warning("No face recognition systems were enabled.")
                
        except Exception as e:
            logging.error(f"Failed to initialize FaceRecognition: {str(e)}")
    else:
        logging.info("Face recognition is disabled.")

    # Initialize video capture
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        logging.error("Failed to open camera!")
        return

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps <= 0:
        actual_fps = FPS
    logging.info(f"Camera FPS: {actual_fps}")

    # Video writer setup for violation recording
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    recording = False
    video_writer = None
    violation_start_time = None
    no_violation_duration = VIDEO_DURATION
    countdown_start_time = None
    output_path = None

    frame_count = 0
    violation_active = False
    last_person_count = 0
    last_phone_detected = False
    last_proctor_violation = False
    last_proctor_text = ""
    last_status_text = "Monitoring normal"
    last_status_color = (0, 255, 0)

    # Face recognition status variables
    face_status = {
        "insightface": {"detected": False, "recognized": False, "match_id": None, "score": -1},
        "dlib": {"detected": False, "recognized": False, "match_id": None, "score": -1}
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to capture frame!")
            break

        frame_count += 1
        current_time = time.time()

        # Process frame
        # Face recognition - separate processing for each system
        if ENABLE_FACE_RECOGNITION:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # InsightFace processing
            if ENABLE_INSIGHTFACE and insightface_recognizer:
                try:
                    embedding, bbox = insightface_recognizer.get_face_embedding(rgb_frame)
                    face_status["insightface"]["detected"] = embedding is not None
                    
                    # Draw face bounding box if detected
                    if bbox is not None:
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    if face_status["insightface"]["detected"]:
                        match_id, similarity = insightface_recognizer.compare_face_with_all(embedding)
                        face_status["insightface"]["recognized"] = match_id is not None and similarity >= FACE_SIMILARITY_THRESHOLD
                        face_status["insightface"]["match_id"] = match_id
                        face_status["insightface"]["score"] = similarity
                    else:
                        face_status["insightface"]["recognized"] = False
                        face_status["insightface"]["match_id"] = None
                        face_status["insightface"]["score"] = -1
                    
                    logging.debug(f"InsightFace: Detected: {face_status['insightface']['detected']}, "
                                 f"Recognized: {face_status['insightface']['recognized']}, "
                                 f"ID: {face_status['insightface']['match_id']}, "
                                 f"Score: {face_status['insightface']['score']:.4f}")
                except Exception as e:
                    logging.error(f"InsightFace processing error: {str(e)}")
                    face_status["insightface"] = {"detected": False, "recognized": False, "match_id": None, "score": -1}

            # Dlib processing
            if ENABLE_DLIB and dlib_recognizer:
                try:
                    embedding, bbox = dlib_recognizer.get_face_embedding(rgb_frame)
                    face_status["dlib"]["detected"] = embedding is not None
                    
                    # Draw face bounding box if detected
                    if bbox is not None:
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    if face_status["dlib"]["detected"]:
                        match_id, distance = dlib_recognizer.compare_face_with_all(embedding)
                        face_status["dlib"]["recognized"] = match_id is not None and distance <= DLIB_DISTANCE_THRESHOLD
                        face_status["dlib"]["match_id"] = match_id
                        face_status["dlib"]["score"] = distance
                    else:
                        face_status["dlib"]["recognized"] = False
                        face_status["dlib"]["match_id"] = None
                        face_status["dlib"]["score"] = -1
                    
                    logging.debug(f"Dlib: Detected: {face_status['dlib']['detected']}, "
                                 f"Recognized: {face_status['dlib']['recognized']}, "
                                 f"ID: {face_status['dlib']['match_id']}, "
                                 f"Distance: {face_status['dlib']['score']:.4f}")
                except Exception as e:
                    logging.error(f"Dlib processing error: {str(e)}")
                    face_status["dlib"] = {"detected": False, "recognized": False, "match_id": None, "score": -1}
        else:
            face_status["insightface"] = {"detected": False, "recognized": False, "match_id": None, "score": -1}
            face_status["dlib"] = {"detected": False, "recognized": False, "match_id": None, "score": -1}

        # Person detection
        if ENABLE_PERSON_DETECTION and person_model:
            try:
                last_person_count, frame = detect_persons(frame, person_model)
            except Exception as e:
                logging.error(f"Person detection error: {str(e)}")
                last_person_count = 0
        else:
            last_person_count = 0
            logging.debug("Person detection is disabled.")

        # Phone detection
        if ENABLE_PHONE_DETECTION and phone_model:
            try:
                last_phone_detected, frame = detect_phone(frame, phone_model)
            except Exception as e:
                logging.error(f"Phone detection error: {str(e)}")
                last_phone_detected = False
        else:
            last_phone_detected = False
            logging.debug("Phone detection is disabled.")

        # Proctoring
        if ENABLE_PROCTORING:
            try:
                frame, last_proctor_violation, last_proctor_text = run_proctoring(frame)
            except Exception as e:
                logging.error(f"Proctoring error: {str(e)}")
                last_proctor_violation = False
                last_proctor_text = ""
        else:
            last_proctor_violation = False
            last_proctor_text = ""
            logging.debug("Proctoring is disabled.")

        # Check for violations
        violation = (
            (ENABLE_FACE_RECOGNITION and (
                (ENABLE_INSIGHTFACE and insightface_recognizer and 
                 not face_status["insightface"]["detected"] or 
                 not face_status["insightface"]["recognized"]) or
                (ENABLE_DLIB and dlib_recognizer and 
                 not face_status["dlib"]["detected"] or 
                 not face_status["dlib"]["recognized"])
            )) or
            (ENABLE_PERSON_DETECTION and last_person_count > 1) or
            (ENABLE_PHONE_DETECTION and last_phone_detected) or
            (ENABLE_PROCTORING and last_proctor_violation)
        )

        # Violation reasons
        violation_reasons = []
        if ENABLE_FACE_RECOGNITION:
            if ENABLE_INSIGHTFACE and insightface_recognizer:
                if not face_status["insightface"]["detected"]:
                    violation_reasons.append("InsightFace: Face not detected")
                elif not face_status["insightface"]["recognized"]:
                    violation_reasons.append("InsightFace: Face not recognized")
            if ENABLE_DLIB and dlib_recognizer:
                if not face_status["dlib"]["detected"]:
                    violation_reasons.append("Dlib: Face not detected")
                elif not face_status["dlib"]["recognized"]:
                    violation_reasons.append("Dlib: Face not recognized")
        if ENABLE_PERSON_DETECTION and last_person_count > 1:
            violation_reasons.append(f"{last_person_count} persons detected")
        if ENABLE_PHONE_DETECTION and last_phone_detected:
            violation_reasons.append("Phone detected")
        if ENABLE_PROCTORING and last_proctor_violation and last_proctor_text:
            violation_reasons.append(last_proctor_text)

        # Update status
        last_status_text = "Monitoring normal" if not violation else "VIOLATION: " + ", ".join(violation_reasons)
        last_status_color = (0, 255, 0) if not violation else (0, 0, 255)

        # Handle violation start
        if violation:
            if not violation_active:
                violation_active = True
                if not recording:
                    # Start video recording
                    timestamp = int(time.time())
                    if not os.path.exists(VIDEO_DIR):
                        try:
                            os.makedirs(VIDEO_DIR)
                            logging.info(f"Created directory: {VIDEO_DIR}")
                        except Exception as e:
                            logging.error(f"Failed to create video directory: {str(e)}")
                    
                    output_path = os.path.join(VIDEO_DIR, f"violation_{timestamp}.avi")
                    height, width = frame.shape[:2]
                    try:
                        video_writer = cv2.VideoWriter(output_path, fourcc, actual_fps, (width, height))
                        if not video_writer.isOpened():
                            logging.error("Failed to initialize video writer!")
                            video_writer = None
                            recording = False
                        else:
                            recording = True
                            violation_start_time = current_time
                            logging.info(f"Violation started at {current_time:.1f}s. Recording started: {output_path}")
                            logging.info(f"Reason: {', '.join(violation_reasons)}")
                    except Exception as e:
                        logging.error(f"Failed to start video recording: {str(e)}")
                        recording = False
                        video_writer = None
                else:
                    logging.info(f"New violation detected at {current_time:.1f}s. Recording continues.")
                    logging.info(f"Reason: {', '.join(violation_reasons)}")
                countdown_start_time = None
        else:
            if violation_active:
                violation_active = False
                countdown_start_time = current_time
                logging.info(f"Violation ended at {current_time:.1f}s. Waiting {no_violation_duration} seconds.")

        # Stop recording
        if recording and not violation_active and countdown_start_time is not None:
            time_since_violation = current_time - countdown_start_time
            if time_since_violation >= no_violation_duration:
                if video_writer is not None:
                    video_writer.release()
                    logging.info(f"Recording stopped at {current_time:.1f}s. Video saved: {output_path}")
                recording = False
                video_writer = None
                violation_start_time = None
                countdown_start_time = None

        # Write frame if recording
        if recording and video_writer is not None:
            try:
                video_writer.write(frame)
            except Exception as e:
                logging.error(f"Failed to write frame to video: {str(e)}")

        # Display status
        cv2.putText(frame, last_status_text, (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, last_status_color, 2)

        # Display face recognition status
        if ENABLE_FACE_RECOGNITION:
            y_offset = 30
            if ENABLE_INSIGHTFACE and insightface_recognizer:
                insightface_text = f"InsightFace: {'ID ' + str(face_status['insightface']['match_id']) if face_status['insightface']['recognized'] else 'Not recognized'} (Score: {face_status['insightface']['score']:.2f})"
                cv2.putText(frame, insightface_text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y_offset += 20
            if ENABLE_DLIB and dlib_recognizer:
                dlib_text = f"Dlib: {'ID ' + str(face_status['dlib']['match_id']) if face_status['dlib']['recognized'] else 'Not recognized'} (Distance: {face_status['dlib']['score']:.2f})"
                cv2.putText(frame, dlib_text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        if recording and not violation_active and countdown_start_time is not None:
            time_since_violation = current_time - countdown_start_time
            remaining = no_violation_duration - time_since_violation
            if remaining > 0:
                cv2.putText(frame, f"Recording ends in: {remaining:.1f}s", (frame.shape[1] - 300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display proctoring violation text if any
        if ENABLE_PROCTORING and last_proctor_violation and last_proctor_text:
            cv2.putText(frame, last_proctor_text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display stream
        cv2.imshow("Real-Time Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    if recording and video_writer is not None:
        video_writer.release()
        logging.info(f"Program stopped. Last video saved: {output_path}")

    cap.release()
    cv2.destroyAllWindows()

def show_main_menu():
    print("\n=== Autoproctoring System ===")
    print("1. Real-time monitoring (face, person, phone, proctoring)")
    print("2. LMDB database management")
    print("0. Exit")
    choice = input("Enter choice (0-2): ")
    return choice

def main():
    # Make sure required directories exist
    for directory in [LMDB_DIR_DLIB, LMDB_DIR_INSIGHTFACE, VIDEO_DIR]:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                logging.info(f"Created directory: {directory}")
            except Exception as e:
                logging.error(f"Failed to create directory {directory}: {str(e)}")

    while True:
        choice = show_main_menu()

        if choice == "1":
            real_time_monitoring()
        elif choice == "2":
            try:
                print("\n=== Database Management ===")
                print("1. Manage Dlib database")
                print("2. Manage InsightFace database")
                print("0. Back to main menu")
                db_choice = input("Enter choice (0-2): ")
                
                if db_choice == "1" and ENABLE_DLIB:
                    db_manager = DBManager(LMDB_DIR_DLIB, model_type="dlib")
                    db_manager.run_menu()
                elif db_choice == "2" and ENABLE_INSIGHTFACE:
                    db_manager = DBManager(LMDB_DIR_INSIGHTFACE, model_type="insightface")
                    db_manager.run_menu()
                elif db_choice == "0":
                    continue
                else:
                    print("Invalid choice or selected recognition system is disabled!")
            except Exception as e:
                logging.error(f"Failed to run DBManager: {str(e)}")
        elif choice == "0":
            print("Exiting system.")
            sys.exit(0)
        else:
            print("Invalid choice! Please enter a number between 0-2.")

if __name__ == "__main__":
    main()