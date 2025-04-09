import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from src.face_recognition import FaceRecognition
from src.db_management import DBManager
from config import *
import time
from collections import deque
from manitoring.proctor import run_proctoring

# Modellarni yuklash
person_model = YOLO(PERSON_MODEL_PATH)
phone_model = YOLO(PHONE_MODEL_PATH)

def detect_persons(frame):
    results = person_model(frame)
    person_count = 0
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 0 and box.conf[0] > DETECTION_CONF_THRESHOLD:
                person_count += 1
    return person_count, frame

def detect_phone(frame):
    results = phone_model(frame)
    phone_detected = False
    for result in results:
        for box in result.boxes:
            if box.conf[0] > DETECTION_CONF_THRESHOLD:
                phone_detected = True
    return phone_detected, frame

def real_time_monitoring():
    face_recognizer = FaceRecognition(LMDB_DIR)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Kamera ochilmadi!")
        return

    actual_fps = 20  # cap.get(cv2.CAP_PROP_FPS) ishlatilishi mumkin
    if actual_fps <= 0:
        actual_fps = FPS
    print(f"Kameraning FPS: {actual_fps}")

    # Video yozuvchisi uchun tayyorgarlik (faqat qoidabuzarlik paytida ishlaydi)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    recording = False
    video_writer = None
    violation_start_time = None
    no_violation_duration = VIDEO_DURATION  # Qoidabuzarlik tugagandan keyin qancha kutish
    countdown_start_time = None

    frame_count = 0
    violation_active = False
    last_person_count = 0
    last_phone_detected = False
    last_proctor_violation = False
    last_proctor_text = ""
    last_face_detected = False
    last_face_recognized = False
    last_status_text = "Nazorat normal"
    last_status_color = (0, 255, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kadr olishda xato!")
            break

        frame_count += 1
        current_time = time.time()

        # Har 4-kadrda tekshirish
        if frame_count % 4 == 0:
            # Yuzni aniqlash
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            embedding, bbox = face_recognizer.get_face_embedding(rgb_frame)
            last_face_detected = embedding is not None
            match_id, similarity = None, -1
            
            if last_face_detected:
                match_id, similarity = face_recognizer.compare_face_with_all(embedding)
                last_face_recognized = match_id is not None and similarity >= FACE_SIMILARITY_THRESHOLD
            else:
                last_face_recognized = False

            # Odam va telefon aniqlash
            last_person_count, frame = detect_persons(frame)
            last_phone_detected, frame = detect_phone(frame)

            # Proctoring qoida buzarligini tekshirish
            frame, last_proctor_violation, last_proctor_text = run_proctoring(frame)

            # Qoidabuzarlikni tekshirish
            violation = (
                not last_face_detected or
                not last_face_recognized or
                last_person_count > 1 or
                last_phone_detected or
                last_proctor_violation
            )

            # Qoida buzilish sabablari
            violation_reasons = []
            if not last_face_detected:
                violation_reasons.append("Yuz aniqlanmadi")
            elif not last_face_recognized:
                violation_reasons.append("Yuz tanilmadi")
            if last_person_count > 1:
                violation_reasons.append(f"{last_person_count} odam aniqlandi")
            if last_phone_detected:
                violation_reasons.append("Telefon aniqlandi")
            if last_proctor_violation and last_proctor_text:
                violation_reasons.append(last_proctor_text)

            # Status yangilash
            last_status_text = "Nazorat normal" if not violation else "QOIDABUZARLIK: " + ", ".join(violation_reasons)
            last_status_color = (0, 255, 0) if not violation else (0, 0, 255)

            # Qoidabuzarlik boshlanishi
            if violation:
                if not violation_active:
                    violation_active = True
                    if not recording:
                        # Video yozishni boshlash
                        timestamp = int(time.time())
                        output_path = os.path.join(VIDEO_DIR, f"violation_{timestamp}.avi")
                        height, width = frame.shape[:2]
                        video_writer = cv2.VideoWriter(output_path, fourcc, actual_fps, (width, height))
                        recording = True
                        violation_start_time = current_time
                        print(f"Qoidabuzarlik boshlandi: {current_time:.1f}s, Yozish boshlandi: {output_path}")
                        print(f"Sabab: {', '.join(violation_reasons)}")
                    else:
                        print(f"Yangi qoidabuzarlik aniqlandi: {current_time:.1f}s, Yozish davom etmoqda")
                        print(f"Sabab: {', '.join(violation_reasons)}")
                    countdown_start_time = None
            else:
                if violation_active:
                    violation_active = False
                    countdown_start_time = current_time
                    print(f"Qoidabuzarlik tugadi: {current_time:.1f}s, {no_violation_duration} sekund kutilmoqda")

            # Yozishni to'xtatish
            if recording and not violation_active and countdown_start_time is not None:
                time_since_violation = current_time - countdown_start_time
                if time_since_violation >= no_violation_duration:
                    video_writer.release()
                    recording = False
                    print(f"Yozish tugadi: {current_time:.1f}s, video saqlandi: {output_path}")
                    video_writer = None
                    violation_start_time = None
                    countdown_start_time = None

        # Agar yozish faol bo'lsa, kadrni yozish
        if recording and video_writer is not None:
            video_writer.write(frame)

        # Har bir kadrda statusni ko'rsatish
        cv2.putText(frame, last_status_text, (10, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, last_status_color, 2)

        if recording and not violation_active and countdown_start_time is not None:
            time_since_violation = current_time - countdown_start_time
            remaining = no_violation_duration - time_since_violation
            if remaining > 0:
                cv2.putText(frame, f"Yozish tugashiga: {remaining:.1f}s", (frame.shape[1] - 300, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Real vaqtda oqimni ko'rsatish
        cv2.imshow("Real-Time Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Agar dastur to'xtasa va yozish davom etayotgan bo'lsa, videoni yopish
    if recording and video_writer is not None:
        video_writer.release()
        print(f"Dastur to'xtadi, oxirgi video saqlandi: {output_path}")

    cap.release()
    cv2.destroyAllWindows()

def show_main_menu():
    print("\n=== Autoproctoring Tizimi ===")
    print("1. Real vaqtda monitoring (yuz, odam, telefon, proctoring)")
    print("2. LMDB baza boshqaruvi")
    print("0. Chiqish")
    choice = input("Tanlovni kiriting (0-2): ")
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
            print("Tizimdan chiqildi.")
            sys.exit(0)
        else:
            print("Noto'g'ri tanlov! Iltimos, 0-2 oralig'ida raqam kiriting.")

if __name__ == "__main__":
    main()