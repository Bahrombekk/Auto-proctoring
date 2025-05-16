# src/monitoring/proctor.py
import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import deque
from config import ENABLE_PROCTORING  # config.py dan ENABLE_PROCTORING ni import qilish

# Mediapipe components
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Key facial landmark indices
NOSE_TIP = 4
RIGHT_EYE = [33, 133]
LEFT_EYE = [362, 263]
MOUTH = [61, 291]
HEAD_POSE_LANDMARKS = [33, 263, 1, 61, 291, 199]

# Qoida buzarlik chegaralari
HEAD_ROTATION_THRESHOLD = 35  # 35% dan ko'p bosh burilishi
SUSTAINED_VIOLATION_TIME = 2.0  # 2 sekund davom etsa
CALIBRATION_FRAMES = 20

# Global o'zgaruvchilar
frame_count = 0
initial_landmarks = None
violation_start_time = None
is_calibrated = False
head_rotation_history = deque(maxlen=10)  # Bosh burilishini filtrlash uchun

# Proctoring funksiyasi
def run_proctoring(frame):
    global initial_landmarks, frame_count, violation_start_time, is_calibrated
    
    # Agar ENABLE_PROCTORING o'chirilgan bo'lsa, proktorlikni ishlatmaslik
    if not ENABLE_PROCTORING:
        return frame, False, "Proctoring is disabled"

    try:
        # Mediapipe face mesh ni sozlash
        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            
            # RGB formatga o'tkazish
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = face_mesh.process(frame_rgb)
            frame_rgb.flags.writeable = True
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            is_violation = False
            violation_text = ""
            head_angle = 0

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    height, width, _ = frame.shape
                    landmarks = face_landmarks.landmark
                    
                    # Kalibratsiya bosqichi
                    if not is_calibrated:
                        # cv2.putText(frame, f"Kalibratsiya qilinyapti... {frame_count}/{CALIBRATION_FRAMES}", 
                        #            (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        if frame_count < CALIBRATION_FRAMES:
                            if frame_count == 0:
                                initial_landmarks = {}
                                for idx in range(len(landmarks)):
                                    initial_landmarks[idx] = [landmarks[idx].x, landmarks[idx].y, landmarks[idx].z]
                            else:
                                for idx in range(len(landmarks)):
                                    if idx in initial_landmarks:
                                        initial_landmarks[idx][0] = 0.9 * initial_landmarks[idx][0] + 0.1 * landmarks[idx].x
                                        initial_landmarks[idx][1] = 0.9 * initial_landmarks[idx][1] + 0.1 * landmarks[idx].y
                                        initial_landmarks[idx][2] = 0.9 * initial_landmarks[idx][2] + 0.1 * landmarks[idx].z
                                    else:
                                        initial_landmarks[idx] = [landmarks[idx].x, landmarks[idx].y, landmarks[idx].z]
                            frame_count += 1
                        else:
                            is_calibrated = True
                            # cv2.putText(frame, "Kalibratsiya tugadi", (30, 90), 
                            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        # Bosh burilishini hisoblash
                        left_eye_x = landmarks[LEFT_EYE[0]].x * width
                        left_eye_y = landmarks[LEFT_EYE[0]].y * height
                        right_eye_x = landmarks[RIGHT_EYE[0]].x * width
                        right_eye_y = landmarks[RIGHT_EYE[0]].y * height
                        
                        eye_distance = math.sqrt((right_eye_x - left_eye_x)**2 + (right_eye_y - left_eye_y)**2)
                        
                        eye_center_x = (right_eye_x + left_eye_x) / 2
                        eye_center_y = (right_eye_y + left_eye_y) / 2
                        
                        nose_x = landmarks[NOSE_TIP].x * width
                        nose_y = landmarks[NOSE_TIP].y * height
                        
                        nose_to_eye_center = math.sqrt((nose_x - eye_center_x)**2 + (nose_y - eye_center_y)**2)
                        
                        rotation_ratio = nose_to_eye_center / eye_distance * 100
                        
                        if LEFT_EYE[0] in initial_landmarks and RIGHT_EYE[0] in initial_landmarks and NOSE_TIP in initial_landmarks:
                            left_cal_x = initial_landmarks[LEFT_EYE[0]][0] * width
                            left_cal_y = initial_landmarks[LEFT_EYE[0]][1] * height
                            right_cal_x = initial_landmarks[RIGHT_EYE[0]][0] * width
                            right_cal_y = initial_landmarks[RIGHT_EYE[0]][1] * height
                            cal_eye_center_x = (right_cal_x + left_cal_x) / 2
                            cal_eye_center_y = (right_cal_y + left_cal_y) / 2
                            cal_nose_x = initial_landmarks[NOSE_TIP][0] * width
                            cal_nose_y = initial_landmarks[NOSE_TIP][1] * height
                            
                            calibration_eye_distance = math.sqrt((right_cal_x - left_cal_x)**2 + (right_cal_y - left_cal_y)**2)
                            calibration_nose_to_eye = math.sqrt((cal_nose_x - cal_eye_center_x)**2 + (cal_nose_y - cal_eye_center_y)**2)
                            
                            if calibration_eye_distance > 0:
                                calibration_ratio = calibration_nose_to_eye / calibration_eye_distance * 100
                                relative_rotation = abs(rotation_ratio - calibration_ratio)
                                head_angle = min(90, relative_rotation * 1.5)
                        
                        head_rotation_history.append(head_angle)
                        filtered_head_angle = sum(head_rotation_history) / len(head_rotation_history)
                        
                        if filtered_head_angle > HEAD_ROTATION_THRESHOLD:
                            if violation_start_time is None:
                                violation_start_time = time.time()
                            
                            violation_duration = time.time() - violation_start_time
                            violation_text = f"Bosh burilishi: {filtered_head_angle:.1f}°"
                            # cv2.putText(frame, violation_text, (width - 350, 60), 
                            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            if violation_duration > SUSTAINED_VIOLATION_TIME:
                                # cv2.putText(frame, "KO'CHIRYAPTI!", (30, 100), 
                                #            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                                # cv2.putText(frame, f"Davomiyligi: {violation_duration:.1f}s", (30, 130), 
                                #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                is_violation = True
                            # else:
                            #     cv2.putText(frame, f"Diqqat! Boshingizni to'g'rilang ({violation_duration:.1f}s/{SUSTAINED_VIOLATION_TIME}s)", 
                            #                (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        else:
                            violation_start_time = None
                        
                        # cv2.putText(frame, f"Bosh burilishi: {filtered_head_angle:.1f}° / {HEAD_ROTATION_THRESHOLD}°", (30, 60), 
                        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            else:
                # cv2.putText(frame, "Yuz topilmadi", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if is_calibrated:
                    if violation_start_time is None:
                        violation_start_time = time.time()
                    
                    violation_duration = time.time() - violation_start_time
                    if violation_duration > SUSTAINED_VIOLATION_TIME:
                        # cv2.putText(frame, "DIQQAT: Talaba kadrdan chiqdi!", (30, 100), 
                        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        # cv2.putText(frame, f"Davomiyligi: {violation_duration:.1f}s", (30, 130), 
                        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        is_violation = True
            
            return frame, is_violation, violation_text
    
    except Exception as e:
        print(f"Error in proctoring: {e}")
        return frame, False, "Error occurred"

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        frame, is_violation, violation_text = run_proctoring(frame)
        cv2.imshow("Proctoring", frame)
        
        if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
            break
    
    cap.release()
    cv2.destroyAllWindows()