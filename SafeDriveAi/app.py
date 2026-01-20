import cv2
import numpy as np
import math
import time
import json
import os


from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from datetime import datetime

# ===============================
# BLACK BOX EVENT STATE
# ===============================
EVENT_COOLDOWN = 8  # seconds

last_event_time = 0
risk_start_time = None

max_yaw = 0
max_pitch = 0
mar_samples = []




BLACKBOX_FILE = "blackbox_log.json"

def log_near_miss(event_type, severity, pitch, yaw, roll, mar):
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event": event_type,
        "severity": severity,
        "pitch": round(pitch, 2),
        "yaw": round(yaw, 2),
        "roll": round(roll, 2),
        "mar": round(mar, 2)
    }

    if not os.path.exists(BLACKBOX_FILE):
        with open(BLACKBOX_FILE, "w") as f:
            json.dump([], f)

    with open(BLACKBOX_FILE, "r+") as f:
        data = json.load(f)
        data.append(log_entry)
        f.seek(0)
        json.dump(data, f, indent=2)

    print("🧾 Near-Miss Logged:", log_entry)



# ======================================================
# HEAD POSE UTILITIES
# ======================================================

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -63.6, -12.5),   # Chin
    (-43.3, 32.7, -26.0),  # Left eye outer
    (43.3, 32.7, -26.0),   # Right eye outer
    (-28.9, -28.9, -24.1), # Left mouth
    (28.9, -28.9, -24.1),  # Right mouth
], dtype=np.float64)

LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

def get_head_pose(face_landmarks, frame_shape):
    h, w = frame_shape[:2]

    image_points = np.array([
        (face_landmarks[i].x * w, face_landmarks[i].y * h)
        for i in LANDMARK_IDS
    ], dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rvec, _ = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rvec)

    pitch = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
    yaw   = math.degrees(math.atan2(-rmat[2, 0],
                                    math.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)))
    roll  = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))

    return pitch, yaw, roll


# ======================================================
# MOUTH / YAWN UTILITIES
# ======================================================

UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 61
RIGHT_MOUTH = 291

def mouth_aspect_ratio(face_landmarks, frame_shape):
    h, w = frame_shape[:2]

    upper = np.array([face_landmarks[UPPER_LIP].x * w,
                      face_landmarks[UPPER_LIP].y * h])
    lower = np.array([face_landmarks[LOWER_LIP].x * w,
                      face_landmarks[LOWER_LIP].y * h])
    left  = np.array([face_landmarks[LEFT_MOUTH].x * w,
                      face_landmarks[LEFT_MOUTH].y * h])
    right = np.array([face_landmarks[RIGHT_MOUTH].x * w,
                      face_landmarks[RIGHT_MOUTH].y * h])

    return np.linalg.norm(upper - lower) / np.linalg.norm(left - right)


# ======================================================
# CALIBRATION & FATIGUE STATE
# ======================================================

CALIBRATION_TIME = 3  # seconds

calibrated = False
calibration_start = None
baseline_pitch = 0
baseline_yaw = 0

pitch_samples = []
yaw_samples = []

YAWN_THRESHOLD = 0.35
YAWN_DURATION = 1.0

yawn_start = None
yawn_events = []


# ======================================================
# MEDIAPIPE FACE LANDMARKER SETUP
# ======================================================

print("📦 Loading Face Landmarker model...")
try:
    base_options = BaseOptions(model_asset_path="face_landmarker.task")

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )

    face_landmarker = vision.FaceLandmarker.create_from_options(options)
    print("✅ Face Landmarker loaded successfully")
except Exception as e:
    print(f"❌ Failed to load Face Landmarker: {e}")
    exit(1)


# ======================================================
# CAMERA
# ======================================================

print("📷 Initializing camera...")
cap = cv2.VideoCapture(0)

# Add a small delay for camera to initialize
time.sleep(1)

if not cap.isOpened():
    print("❌ Camera not accessible")
    print("   Possible reasons:")
    print("   1. Camera is already in use by another app")
    print("   2. Camera permissions not granted (macOS: Settings → Security & Privacy → Camera)")
    print("   3. No camera device found")
    exit(1)

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("✅ Camera initialized successfully")


# ======================================================
# MAIN LOOP
# ======================================================

# ===============================
# NEAR-MISS STATE
# ===============================
distraction_frames = 0


while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)

    result = face_landmarker.detect(mp_img)

    if result.face_landmarks:
        face_landmarks = result.face_landmarks[0]

        # Draw mesh
        for lm in face_landmarks:
            x = int(lm.x * frame.shape[1])
            y = int(lm.y * frame.shape[0])
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        pose = get_head_pose(face_landmarks, frame.shape)
        if pose:
            pitch, yaw, roll = pose
            now = time.time()

            # ---------------- CALIBRATION ----------------
            if not calibrated:
                if calibration_start is None:
                    calibration_start = now

                pitch_samples.append(pitch)
                yaw_samples.append(yaw)

                remaining = max(0, int(CALIBRATION_TIME - (now - calibration_start)))

                cv2.putText(frame,
                            f"CALIBRATING... LOOK STRAIGHT ({remaining}s)",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 255), 2)

                if now - calibration_start >= CALIBRATION_TIME:
                    baseline_pitch = np.mean(pitch_samples)
                    baseline_yaw = np.mean(yaw_samples)
                    calibrated = True
                    print("✅ Calibration complete")

            # ---------------- NORMAL MODE ----------------
            else:
                pitch_delta = pitch - baseline_pitch
                yaw_delta = yaw - baseline_yaw
                mar = mouth_aspect_ratio(face_landmarks, frame.shape)

                danger = False
                reason = None

                if pitch_delta < -12:
                    danger = True
                    reason = "PHONE_USAGE"
                elif abs(yaw_delta) > 18:
                    danger = True
                    reason = "SIDE_DISTRACTION"

                # ===============================
                # NEAR-MISS BLACK BOX LOGIC
                # ===============================
                if danger:
                    if risk_start_time is None:
                        risk_start_time = now

                    max_pitch = min(max_pitch, pitch_delta)
                    max_yaw = max(max_yaw, abs(yaw_delta))
                    mar_samples.append(mar)

                    duration = now - risk_start_time

                    if duration >= 3:
                        status = "🚨 HIGH RISK"
                        color = (0, 0, 255)
                        severity = "HIGH"
                    else:
                        status = "⚠️ DISTRACTED"
                        color = (0, 165, 255)
                        severity = "MEDIUM"

                else:
                    # Driver recovered → log once
                    if risk_start_time:
                        duration = now - risk_start_time

                        if duration >= 1.5 and now - last_event_time >= EVENT_COOLDOWN:
                            log_near_miss(
                                event_type="NEAR_MISS",
                                severity=severity,
                                pitch=max_pitch,
                                yaw=max_yaw,
                                roll=roll,
                                mar=float(np.mean(mar_samples)) if mar_samples else 0
                            )
                            last_event_time = now

                        # Reset black-box state
                        risk_start_time = None
                        max_pitch = 0
                        max_yaw = 0
                        mar_samples.clear()

                    status = "✅ ATTENTIVE"
                    color = (0, 255, 0)

                cv2.putText(frame, status, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

                # ===============================
                # FATIGUE / YAWN WARNING (NON-LOGGING)
                # ===============================
                if mar > YAWN_THRESHOLD:
                    if yawn_start is None:
                        yawn_start = now
                else:
                    if yawn_start and now - yawn_start >= YAWN_DURATION:
                        yawn_events.append(now)
                    yawn_start = None

                yawn_events = [t for t in yawn_events if now - t <= 120]

                if len(yawn_events) >= 3:
                    cv2.putText(frame,
                                "⚠️ FATIGUE DETECTED – TAKE A BREAK",
                                (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 255, 255), 3)
    cv2.imshow("SafeDrive-AI", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): 
        break

cap.release()
cv2.destroyAllWindows()

#hu 
#hu 
#hu 
#hu 
