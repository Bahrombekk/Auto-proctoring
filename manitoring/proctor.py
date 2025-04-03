import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import deque

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
HEAD_ROTATION_THRESHOLD = 35  # 45% dan ko'p bosh burilishi
SUSTAINED_VIOLATION_TIME = 2.0  # 10 sekund davom etsa
CALIBRATION_FRAMES = 20

# Global o'zgaruvchilar
frame_count = 0
initial_landmarks = None
violation_start_time = None
is_calibrated = False
head_rotation_history = deque(maxlen=10)  # Bosh burilishini filtrlash uchun

# Proctoring funksiyasi (frame qabul qilib, qoida buzarlikni qaytaradi)
def run_proctoring(frame):
    global initial_landmarks, frame_count, violation_start_time, is_calibrated
    
    # Mediapipe face mesh ni sozlash
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        
        # RGB formatga o'tkazish
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = face_mesh.process(frame_rgb)
        frame.flags.writeable = True
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
                    #cv2.putText(frame, f"Kalibratsiya qilinyapti... {frame_count}/{CALIBRATION_FRAMES}", 
                    #           (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if frame_count < CALIBRATION_FRAMES:
                        if frame_count == 0:
                            # Kalibratsiya uchun har bir nuqtani saqlash 
                            initial_landmarks = {}
                            for idx in range(len(landmarks)):
                                initial_landmarks[idx] = [landmarks[idx].x, landmarks[idx].y, landmarks[idx].z]
                        else:
                            # Yumshoqroq kalibratsiya
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
                        #cv2.putText(frame, "Kalibratsiya tugadi", (30, 90), 
                        #          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Bosh burilishini hisoblash
                    # 1. O'ng va chap ko'z nuqtalarini aniqlash
                    left_eye_x = landmarks[LEFT_EYE[0]].x * width
                    left_eye_y = landmarks[LEFT_EYE[0]].y * height
                    right_eye_x = landmarks[RIGHT_EYE[0]].x * width
                    right_eye_y = landmarks[RIGHT_EYE[0]].y * height
                    
                    # 2. Ko'zlar orasidagi masofani hisoblash
                    eye_distance = math.sqrt((right_eye_x - left_eye_x)**2 + (right_eye_y - left_eye_y)**2)
                    
                    # 3. Ko'zlar markazini hisoblash
                    eye_center_x = (right_eye_x + left_eye_x) / 2
                    eye_center_y = (right_eye_y + left_eye_y) / 2
                    
                    # 4. Burun nuqtasini aniqlash
                    nose_x = landmarks[NOSE_TIP].x * width
                    nose_y = landmarks[NOSE_TIP].y * height
                    
                    # 5. Burun va ko'zlar markazi orasidagi masofani hisoblash
                    nose_to_eye_center = math.sqrt((nose_x - eye_center_x)**2 + (nose_y - eye_center_y)**2)
                    
                    # 6. Bosh burilishini hisoblash - burchak o'rniga nisbiy masofani ishlatamiz
                    # Markaziy pozitsiyada burun va ko'zlar markazi orasidagi masofa eng qisqa bo'ladi
                    # Bosh burilganda bu masofa ko'payadi
                    rotation_ratio = nose_to_eye_center / eye_distance * 100
                    
                    # Kalibratsiya vaqtidagi qiymatni hisobga olish
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
                        
                        if calibration_eye_distance > 0:  # Nolga bo'lishdan himoya
                            calibration_ratio = calibration_nose_to_eye / calibration_eye_distance * 100
                            
                            # Nisbiy o'zgarish - kalibratsiya qiymatiga nisbatan
                            relative_rotation = abs(rotation_ratio - calibration_ratio)
                            
                            # Bosh burilishi burchagi - taxminiy hisoblash (45 darajaga to'g'rilangan)
                            head_angle = min(90, relative_rotation * 1.5)  # Ko'paytiruvchi 1.5 tajriba orqali tanlanadi
                    else:
                        # Agar kalibratsiya ma'lumotlari bo'lmasa
                        head_angle = 0
                    
                    # Filtrlash
                    head_rotation_history.append(head_angle)
                    filtered_head_angle = sum(head_rotation_history) / len(head_rotation_history)
                    
                    # Qoida buzarlikni tekshirish
                    if filtered_head_angle > HEAD_ROTATION_THRESHOLD:
                        # Buzilish boshlangan vaqtni belgilash
                        if violation_start_time is None:
                            violation_start_time = time.time()
                        
                        # Buzilish davomiyligi
                        violation_duration = time.time() - violation_start_time
                        
                        # Qoida buzarlik belgilari
                        violation_text = f"Bosh burilishi: {filtered_head_angle:.1f}°"
                        #cv2.putText(frame, violation_text, (width - 350, 60), 
                        #          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Uzoq davom etgan qoida buzarlik
                        if violation_duration > SUSTAINED_VIOLATION_TIME:
                        #    cv2.putText(frame, "KO'CHIRYAPTI!", (30, 100), 
                        #              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        #    cv2.putText(frame, f"Davomiyligi: {violation_duration:.1f}s", (30, 130), 
                        #              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            is_violation = True
                        #else:
                            # Ogohlantirish
                            #cv2.putText(frame, f"Diqqat! Boshingizni to'g'rilang ({violation_duration:.1f}s/{SUSTAINED_VIOLATION_TIME}s)", 
                            #          (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    else:
                        violation_start_time = None
                    
                    # Vizualizatsiya
                    #cv2.putText(frame, f"Bosh burilishi: {filtered_head_angle:.1f}° / {HEAD_ROTATION_THRESHOLD}°", (30, 60), 
                    #          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    # Yuz nuqtalarini chizish
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
        else:
            #cv2.putText(frame, "Yuz topilmadi", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Yuz yo'qolganligini alohida tekshirish
            if is_calibrated:
                if violation_start_time is None:
                    violation_start_time = time.time()
                
                violation_duration = time.time() - violation_start_time
                # Faqat uzoq vaqt yuz ko'rinmasa buzilish sifatida qayd etish
                if violation_duration > SUSTAINED_VIOLATION_TIME:
                    #cv2.putText(frame, "DIQQAT: Talaba kadrdan chiqdi!", (30, 100), 
                    #          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    #cv2.putText(frame, f"Davomiyligi: {violation_duration:.1f}s", (30, 130), 
                    #          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    is_violation = True
                
        return frame, is_violation, violation_text

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame, is_violation, violation_text = run_proctoring(frame)
        cv2.imshow("Proctoring", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()