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
person_model = YOLO(PERSON_MODEL_PATH) if ENABLE_PERSON_DETECTION else None
phone_model = YOLO(PHONE_MODEL_PATH) if ENABLE_PHONE_DETECTION else None

def real_time_monitoring():
    # Initialize face recognition if enabled
    if ENABLE_FACE_RECOGNITION:
        face_recognizer = FaceRecognition(LMDB_DIR)
    else:
        logging.info("Face recognition is disabled.")
        face_recognizer = None

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

    frame_count = 0
    violation_active = False
    last_person_count = 0
    last_phone_detected = False
    last_proctor_violation = False
    last_proctor_text = ""
    last_face_detected = False
    last_face_recognized = False
    last_status_text = "Monitoring normal"
    last_status_color = (0, 255, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to capture frame!")
            break

        frame_count += 1
        current_time = time.time()

        # Process frame
        # Face recognition
        if ENABLE_FACE_RECOGNITION and face_recognizer:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            embedding, bbox = face_recognizer.get_face_embedding(rgb_frame)
            last_face_detected = embedding is not None
            match_id, similarity = None, -1

            if last_face_detected:
                match_id, similarity = face_recognizer.compare_face_with_all(embedding)
                last_face_recognized = match_id is not None and similarity >= FACE_SIMILARITY_THRESHOLD
            else:
                last_face_recognized = False
        else:
            last_face_detected = False
            last_face_recognized = False

        # Person detection
        if ENABLE_PERSON_DETECTION and person_model:
            last_person_count, frame = detect_persons(frame, person_model)
        else:
            last_person_count = 0
            logging.debug("Person detection is disabled.")

        # Phone detection
        if ENABLE_PHONE_DETECTION and phone_model:
            last_phone_detected, frame = detect_phone(frame, phone_model)
        else:
            last_phone_detected = False
            logging.debug("Phone detection is disabled.")

        # Proctoring
        if ENABLE_PROCTORING:
            frame, last_proctor_violation, last_proctor_text = run_proctoring(frame)
        else:
            last_proctor_violation = False
            last_proctor_text = ""
            logging.debug("Proctoring is disabled.")

        # Check for violations
        violation = (
            (ENABLE_FACE_RECOGNITION and (not last_face_detected or not last_face_recognized)) or
            (ENABLE_PERSON_DETECTION and last_person_count > 1) or
            (ENABLE_PHONE_DETECTION and last_phone_detected) or
            (ENABLE_PROCTORING and last_proctor_violation)
        )

        # Violation reasons
        violation_reasons = []
        if ENABLE_FACE_RECOGNITION:
            if not last_face_detected:
                violation_reasons.append("Face not detected")
            elif not last_face_recognized:
                violation_reasons.append("Face not recognized")
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
                    output_path = os.path.join(VIDEO_DIR, f"violation_{timestamp}.avi")
                    height, width = frame.shape[:2]
                    video_writer = cv2.VideoWriter(output_path, fourcc, actual_fps, (width, height))
                    recording = True
                    violation_start_time = current_time
                    logging.info(f"Violation started at {current_time:.1f}s. Recording started: {output_path}")
                    logging.info(f"Reason: {', '.join(violation_reasons)}")
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
                video_writer.release()
                recording = False
                logging.info(f"Recording stopped at {current_time:.1f}s. Video saved: {output_path}")
                video_writer = None
                violation_start_time = None
                countdown_start_time = None

        # Write frame if recording
        if recording and video_writer is not None:
            video_writer.write(frame)

        # Display status
        cv2.putText(frame, last_status_text, (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, last_status_color, 2)

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
    while True:
        choice = show_main_menu()

        if choice == "1":
            real_time_monitoring()
        elif choice == "2":
            db_manager = DBManager(LMDB_DIR)
            db_manager.run_menu()
        elif choice == "0":
            print("Exiting system.")
            sys.exit(0)
        else:
            print("Invalid choice! Please enter a number between 0-2.")

if __name__ == "__main__":
    main()