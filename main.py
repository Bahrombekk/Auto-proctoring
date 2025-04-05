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
from manitoring.proctor import run_proctoring  # Proctoring funksiyasini import qilamiz

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
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #cv2.putText(frame, f"Person {box.conf[0]:.2f}", (x1, y1 - 10),
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return person_count, frame

def detect_phone(frame):
    results = phone_model(frame)
    phone_detected = False
    for result in results:
        for box in result.boxes:
            if box.conf[0] > DETECTION_CONF_THRESHOLD:
                phone_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                #cv2.putText(frame, f"Phone {box.conf[0]:.2f}", (x1, y1 - 10),
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return phone_detected, frame

def save_video(frame_buffer, output_path, fps, start_time, end_time):
    
    if not frame_buffer:
        print("Bufer bo'sh, video saqlanmadi.")
        return
    
    height, width = frame_buffer[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frame_buffer:
        out.write(frame)
    out.release()
    print(f"Video saqlandi: {output_path} ({start_time:.1f}s - {end_time:.1f}s, FPS: {fps})")

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

    max_buffer_size = int(actual_fps * VIDEO_DURATION * 2)
    frame_buffer = deque(maxlen=max_buffer_size)
    timestamps = deque(maxlen=max_buffer_size)

    recording = False
    violation_active = False
    violation_start_time = None
    last_violation_time = None
    countdown_start_time = None
    no_violation_duration = VIDEO_DURATION

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kadr olishda xato!")
            break

        frame_count += 1
        current_time = time.time()
        frame_buffer.append(frame.copy())
        timestamps.append(current_time)

        # Yuzni aniqlash
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        embedding, bbox = face_recognizer.get_face_embedding(rgb_frame)
        face_detected = embedding is not None
        match_id, similarity = None, -1
        
        if face_detected:
            match_id, similarity = face_recognizer.compare_face_with_all(embedding)
            face_recognized = match_id is not None and similarity >= FACE_SIMILARITY_THRESHOLD
        else:
            face_recognized = False

        # Odam va telefon aniqlash
        person_count, frame = detect_persons(frame)
        phone_detected, frame = detect_phone(frame)

        # Proctoring qoida buzarligini tekshirish
        frame, proctor_violation, proctor_violation_text = run_proctoring(frame)

        # Yuz vizualizatsiyasi
        if face_detected and bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            if face_recognized:
                label = f"ID: {match_id} ({similarity:.2f})"
                color = (0, 255, 0)
            else:
                label = f"Noma'lum ({similarity:.2f})"
                color = (0, 0, 255)
            #cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            #cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        #else:
            #cv2.putText(frame, "Yuz topilmadi", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Qoidabuzarlikni tekshirish (proctor qoida buzarligi qo'shildi)
        violation = (
            not face_detected or
            not face_recognized or
            person_count > 1 or
            phone_detected or
            proctor_violation  # Proctoringdan kelgan qoida buzarlik
        )

        # Qoida buzilish sabablari
        violation_reasons = []
        if not face_detected:
            violation_reasons.append("Yuz aniqlanmadi")
        elif not face_recognized:
            violation_reasons.append("Yuz tanilmadi")
        if person_count > 1:
            violation_reasons.append(f"{person_count} odam aniqlandi")
        if phone_detected:
            violation_reasons.append("Telefon aniqlandi")
        if proctor_violation and proctor_violation_text:
            violation_reasons.append(proctor_violation_text)  # Proctoring sababi

        # Status ko'rsatish
        status_text = "Nazorat normal" if not violation else "QOIDABUZARLIK: " + ", ".join(violation_reasons)
        status_color = (0, 255, 0) if not violation else (0, 0, 255)
        cv2.putText(frame, status_text, (10, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        #if recording:
            #cv2.putText(frame, "YOZILMOQDA", (frame.shape[1] - 200, frame.shape[0] - 20), 
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        # Qoidabuzarlik boshlanishi
        if violation:
            if not violation_active:
                violation_active = True
                last_violation_time = current_time
                if not recording:
                    recording = True
                    first_timestamp_idx = max(0, len(timestamps) - int(actual_fps * VIDEO_DURATION))
                    violation_start_time = timestamps[first_timestamp_idx] if timestamps else current_time
                    print(f"Qoidabuzarlik boshlandi: {current_time:.1f}s, Yozish boshlandi: {violation_start_time:.1f}s")
                    print(f"Sabab: {', '.join(violation_reasons)}")
                else:
                    print(f"Yangi qoidabuzarlik aniqlandi: {current_time:.1f}s, Yozish davom etmoqda")
                    print(f"Sabab: {', '.join(violation_reasons)}")
                countdown_start_time = None
            else:
                last_violation_time = current_time
        else:
            if violation_active:
                violation_active = False
                countdown_start_time = current_time
                print(f"Qoidabuzarlik tugadi: {current_time:.1f}s, {no_violation_duration} sekund kutilmoqda")

        # Yozishni to'xtatish
        if recording and not violation_active and countdown_start_time is not None:
            time_since_violation = current_time - countdown_start_time
            remaining = no_violation_duration - time_since_violation
            if remaining > 0:
                cv2.putText(frame, f"Yozish tugashiga: {remaining:.1f}s", (frame.shape[1] - 300, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if time_since_violation >= no_violation_duration:
                timestamp = int(time.time())
                output_path = os.path.join(VIDEO_DIR, f"violation_{timestamp}.avi")
                save_video(list(frame_buffer), output_path, actual_fps, violation_start_time, current_time)
                recording = False
                violation_start_time = None
                countdown_start_time = None
                print(f"Yozish tugadi: {current_time:.1f}s, video saqlandi")

        cv2.imshow("Real-Time Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if recording:
        timestamp = int(time.time())
        output_path = os.path.join(VIDEO_DIR, f"violation_{timestamp}.avi")
        save_video(list(frame_buffer), output_path, actual_fps, violation_start_time, current_time)
        print("Dastur to'xtadi, oxirgi video saqlandi.")

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
