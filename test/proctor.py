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
RIGHT_EYE = [33, 133]  # Inner and outer eye corners
LEFT_EYE = [362, 263]  # Inner and outer eye corners
MOUTH = [61, 291]  # Left and right mouth corners
HEAD_POSE_LANDMARKS = [33, 263, 1, 61, 291, 199]  # Key points for head pose

# QIYMATLAR OSHIRILDI - harakatlar sezgirligini kamaytirish uchun
LOOKING_AWAY_THRESHOLD = 40  # 40 dan 60 ga oshirildi - burilishga ko'proq yo'l qo'yiladi
MOVEMENT_THRESHOLD = 40      # 40 dan 60 ga oshirildi - harakatga ko'proq yo'l qo'yiladi
SUSTAINED_VIOLATION_TIME = 5.0  # Xatolikni qayd etish uchun vaqt
CALIBRATION_FRAMES = 50     # Kalibrlash uchun kadrlar soni

# Kichik harakatlarni e'tiborsiz qoldirish uchun minimal chegara
MOVEMENT_DEADZONE = 15

# Initialize variables
frame_count = 0
initial_landmarks = None
violations_history = deque(maxlen=120)  # 60 dan 120 ga oshirildi - uzoqroq tarix
violation_start_time = None
is_calibrated = False

# Ko'zning EAR qiymati
def calculate_ear(eye_landmarks, landmarks, image_shape):
    height, width = image_shape[:2]
    
    # Get coordinates of eye landmarks
    coords = []
    for point in eye_landmarks:
        x = int(landmarks[point].x * width)
        y = int(landmarks[point].y * height)
        coords.append((x, y))
    
    # EAR hisoblash uchun yaxshilashtirilgan
    eye_width = np.linalg.norm(np.array(coords[0]) - np.array(coords[1]))
    return eye_width / width  # Normalized by face width

# Harakatni filtrlash uchun yangi o'zgaruvchilar
last_movements = deque(maxlen=10)  # 5 dan 10 ga oshirildi - ko'proq smoothing
filtered_movement = 0

# Harakat yo'nalishlarini kuzatish uchun
last_positions = deque(maxlen=20)  # So'nggi 20 ta holat
direction_changes = 0

# Main function
def run_proctoring():
    global initial_landmarks, frame_count, violation_start_time, is_calibrated, filtered_movement, direction_changes
    
    # Kamerani ochish
    cap = cv2.VideoCapture(0)
    
    # Mediapipe face mesh ni sozlash
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        last_nose_x = None
        last_direction = 0  # Boshlang'ich qiymat 0
        stability_counter = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Kamerani o'qishda xatolik.")
                break

            # RGB formatga o'tkazish
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = face_mesh.process(frame_rgb)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Yuzni aniqlash
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    height, width, _ = frame.shape
                    landmarks = face_landmarks.landmark
                    
                    # Kalibratsiya bosqichi
                    if not is_calibrated:
                        cv2.putText(frame, f"Kalibratsiya qilinyapti... {frame_count}/{CALIBRATION_FRAMES}", 
                                   (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        if frame_count < CALIBRATION_FRAMES:
                            if frame_count == 0:
                                initial_landmarks = {}
                                for idx in HEAD_POSE_LANDMARKS:
                                    initial_landmarks[idx] = [landmarks[idx].x, landmarks[idx].y, landmarks[idx].z]
                            # Kalibratsiya paytida ham boshlang'ich ma'lumotlarni yangilab borish
                            elif frame_count % 10 == 0:  # Har 10 kadrda yangilash
                                for idx in HEAD_POSE_LANDMARKS:
                                    initial_landmarks[idx][0] = 0.9 * initial_landmarks[idx][0] + 0.1 * landmarks[idx].x
                                    initial_landmarks[idx][1] = 0.9 * initial_landmarks[idx][1] + 0.1 * landmarks[idx].y
                                    initial_landmarks[idx][2] = 0.9 * initial_landmarks[idx][2] + 0.1 * landmarks[idx].z
                            frame_count += 1
                        else:
                            is_calibrated = True
                            print("Kalibratsiya tugadi. Kuzatuv boshlandi.")
                        
                    else:
                        # Yuz belgilarini olish
                        current_nose = [landmarks[NOSE_TIP].x * width, landmarks[NOSE_TIP].y * height]
                        
                        # Yo'nalish o'zgarishlarini kuzatish uchun
                        current_nose_x = landmarks[NOSE_TIP].x
                        current_direction = 0  # Har doim qiymatni aniqlash
                        
                        if last_nose_x is not None:
                            current_direction = 1 if current_nose_x > last_nose_x else (-1 if current_nose_x < last_nose_x else 0)
                            
                            if last_direction != 0 and current_direction != 0 and current_direction != last_direction:
                                direction_changes += 1
                            
                            # Yo'nalish o'zgarishlarini vaqt bilan kamaytirish
                            if frame_count % 30 == 0 and direction_changes > 0:
                                direction_changes -= 1
                                
                        last_nose_x = current_nose_x
                        last_direction = current_direction if current_direction != 0 else last_direction
                        
                        # Bosh burilishi va o'rnini o'zgartirishni hisoblash
                        head_rotation = 0
                        head_displacement = 0
                        
                        for idx in HEAD_POSE_LANDMARKS:
                            current_x = landmarks[idx].x
                            current_y = landmarks[idx].y
                            current_z = landmarks[idx].z
                            
                            initial_x = initial_landmarks[idx][0]
                            initial_y = initial_landmarks[idx][1]
                            initial_z = initial_landmarks[idx][2]
                            
                            # Siljish hisoblash
                            disp_x = abs(current_x - initial_x) * width
                            disp_y = abs(current_y - initial_y) * height
                            disp = math.sqrt(disp_x**2 + disp_y**2)
                            head_displacement += disp
                            
                            # Burilishni hisoblash (takomillashtirilgan)
                            z_diff = (current_z - initial_z) * 100
                            head_rotation += abs(z_diff)
                        
                        # Belgilar bo'yicha o'rtacha qiymat
                        head_displacement /= len(HEAD_POSE_LANDMARKS)
                        head_rotation /= len(HEAD_POSE_LANDMARKS)
                        
                        # Harakatlarni saqlash
                        last_positions.append([landmarks[NOSE_TIP].x, landmarks[NOSE_TIP].y, landmarks[NOSE_TIP].z])
                        
                        # Harakatni filtrlash - tebranishlarni yo'qotish uchun
                        last_movements.append(head_displacement)
                        filtered_movement = sum(last_movements) / len(last_movements)
                        
                        # Kichik harakatlarni e'tiborsiz qoldirish
                        if filtered_movement < MOVEMENT_DEADZONE:
                            filtered_movement = 0
                            stability_counter += 1
                        else:
                            stability_counter = 0
                        
                        # Ko'z harakatlarini tekshirish
                        left_ear = calculate_ear(LEFT_EYE, landmarks, frame.shape)
                        right_ear = calculate_ear(RIGHT_EYE, landmarks, frame.shape)
                        
                        # Xatoliklarni tekshirish - bosh harakati, burilish, ko'z qarashlari
                        is_violation = False
                        violation_text = ""
                        violation_severity = 0  # Yangi: xatolik darajasi (0-10)
                        
                        # Harakatni tekshirish - endi filtrlangan qiymat bilan
                        if filtered_movement > MOVEMENT_THRESHOLD:
                            # Ko'p yo'nalish o'zgarishlari bo'lsagina, shubhali deb belgilash
                            if direction_changes > 4:  # Tez-tez yo'nalish o'zgarishi - shubhali
                                movement_severity = min(10, int((filtered_movement - MOVEMENT_THRESHOLD) / 5) + 1)
                                is_violation = True
                                violation_severity = max(violation_severity, movement_severity)
                                violation_text = f"Bosh harakati! ({movement_severity}/10)"
                        
                        # Burilishni tekshirish
                        if head_rotation > LOOKING_AWAY_THRESHOLD:
                            rotation_severity = min(10, int((head_rotation - LOOKING_AWAY_THRESHOLD) / 5) + 1)
                            is_violation = True
                            violation_severity = max(violation_severity, rotation_severity)
                            violation_text = f"Tomon o'girish! ({rotation_severity}/10)"
                        
                        # Xatoliklar tarixini yangilash
                        violations_history.append(is_violation)
                        
                        # Doimiy xatolikni tekshirish - endi darajani hisobga olgan holda
                        if is_violation:
                            if violation_start_time is None:
                                violation_start_time = time.time()
                            # Xatolik darajasiga qarab, shubhali vaqt kamaytiriladi
                            required_time = max(1.0, SUSTAINED_VIOLATION_TIME - (violation_severity * 0.2))
                            if time.time() - violation_start_time > required_time:
                                cv2.putText(frame, "KO'CHIRYAPTI!", (30, 100), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                                # Qo'shimcha: jiddiy buzilish darajasini ko'rsatish
                                cv2.putText(frame, f"Daraja: {violation_severity}/10", (30, 130), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        else:
                            violation_start_time = None
                        
                        # Joriy qiymatlarni ko'rsatish
                        cv2.putText(frame, f"Harakat: {filtered_movement:.1f}/{MOVEMENT_THRESHOLD}", (30, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        cv2.putText(frame, f"Burilish: {head_rotation:.1f}/{LOOKING_AWAY_THRESHOLD}", (30, 85), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        cv2.putText(frame, f"Yo'nalish o'zgarishlari: {direction_changes}", (30, 110), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        
                        if violation_text:
                            cv2.putText(frame, violation_text, (width - 350, 60), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Shubhali xatti-harakatlar foizini hisoblash
                        violation_percentage = sum(violations_history) / len(violations_history) * 100 if violations_history else 0
                        cv2.putText(frame, f"Shubhali harakatlar: {violation_percentage:.1f}%", 
                                  (30, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        # Moslashuvchan referens pozitsiya yangilash
                        # 1. Agar uzluksiz barqarorlik bo'lsa (kichik harakatlar)
                        if stability_counter > 50:  # 50 kadr davomida barqaror
                            stability_counter = 0
                            adaptation_rate = 0.1  # 10% moslashuv
                            for idx in HEAD_POSE_LANDMARKS:
                                initial_landmarks[idx][0] = (1-adaptation_rate) * initial_landmarks[idx][0] + adaptation_rate * landmarks[idx].x
                                initial_landmarks[idx][1] = (1-adaptation_rate) * initial_landmarks[idx][1] + adaptation_rate * landmarks[idx].y
                                initial_landmarks[idx][2] = (1-adaptation_rate) * initial_landmarks[idx][2] + adaptation_rate * landmarks[idx].z
                            print("Boshlang'ich holat yangilandi - barqarorlik asosida")
                        
                        # 2. Vaqti-vaqti bilan kichik moslashuv
                        if not is_violation and frame_count % 100 == 0:  # Har 100 kadrda
                            adaptation_rate = 0.05  # 5% moslashuv
                            for idx in HEAD_POSE_LANDMARKS:
                                initial_landmarks[idx][0] = (1-adaptation_rate) * initial_landmarks[idx][0] + adaptation_rate * landmarks[idx].x
                                initial_landmarks[idx][1] = (1-adaptation_rate) * initial_landmarks[idx][1] + adaptation_rate * landmarks[idx].y
                                initial_landmarks[idx][2] = (1-adaptation_rate) * initial_landmarks[idx][2] + adaptation_rate * landmarks[idx].z
                        
                        # 3. Uzoq vaqt davomida katta moslashuv - talabani qimirlamaslikka majbur qilmaslik uchun
                        if violation_percentage < 15 and frame_count % 300 == 0:  # Har 300 kadrda
                            adaptation_rate = 0.2  # 20% moslashuv
                            for idx in HEAD_POSE_LANDMARKS:
                                initial_landmarks[idx][0] = (1-adaptation_rate) * initial_landmarks[idx][0] + adaptation_rate * landmarks[idx].x
                                initial_landmarks[idx][1] = (1-adaptation_rate) * initial_landmarks[idx][1] + adaptation_rate * landmarks[idx].y
                                initial_landmarks[idx][2] = (1-adaptation_rate) * initial_landmarks[idx][2] + adaptation_rate * landmarks[idx].z
                            print("Boshlang'ich holat yangilandi - uzoq muddatli adaptatsiya")
                    
                    # Yuz nuqtalarini chizish
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            else:
                cv2.putText(frame, "Yuz topilmadi", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Yuz yo'qolganda kalibrovkani tiklash
                if is_calibrated:
                    cv2.putText(frame, "DIQQAT: Talaba kadrdan chiqdi!", (30, 100), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Natijani ko'rsatish
            cv2.imshow("Avtoproktoring tizimi", frame)
            frame_count += 1

            # ESC tugmasi bilan chiqish
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_proctoring()