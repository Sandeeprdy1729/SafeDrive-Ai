import cv2
import numpy as np
import math

from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions


# ===============================
# HEAD POSE UTILS 
# ===============================

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

    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rvec)

    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
        yaw   = math.degrees(math.atan2(-rmat[2, 0], sy))
        roll  = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
    else:
        pitch = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
        yaw   = math.degrees(math.atan2(-rmat[2, 0], sy))
        roll  = 0

    return pitch, yaw, roll

# ===============================
# CALIBRATION STATE
# ===============================
CALIBRATION_TIME = 3  # seconds

calibrated = False
calibration_start = None
baseline_pitch = 0
baseline_yaw = 0

pitch_samples = []
yaw_samples = []


# -------------------------------
# FACE LANDMARKER SETUP
# -------------------------------
base_options = BaseOptions(model_asset_path="face_landmarker.task")

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)

face_landmarker = vision.FaceLandmarker.create_from_options(options)

# -------------------------------
# CAMERA
# -------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not accessible. Check macOS permissions.")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_img = Image(
        image_format=ImageFormat.SRGB,
        data=rgb
    )

    result = face_landmarker.detect(mp_img)

    if result.face_landmarks:
        # ✅ DEFINE IT FIRST
        face_landmarks = result.face_landmarks[0]

        # 1️⃣ Draw landmarks
        for lm in face_landmarks:
            x = int(lm.x * frame.shape[1])
            y = int(lm.y * frame.shape[0])
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # 2️⃣ Head pose
        pose = get_head_pose(face_landmarks, frame.shape)
        if pose:
            pitch, yaw, roll = pose

            cv2.putText(frame, f"Pitch: {pitch:.1f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Yaw: {yaw:.1f}", (20, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Roll: {roll:.1f}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 3️⃣ Intelligent status (USP)
            current_time = cv2.getTickCount() / cv2.getTickFrequency()

            # -------------------------------
            # CALIBRATION PHASE (EXCLUSIVE)
            # -------------------------------
            if not calibrated:
                if calibration_start is None:
                    calibration_start = current_time

                pitch_samples.append(pitch)
                yaw_samples.append(yaw)

                elapsed = current_time - calibration_start

                cv2.putText(frame,
                            f"CALIBRATING... LOOK STRAIGHT ({int(CALIBRATION_TIME - elapsed)}s)",
                            (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2)

                # ❗ STOP processing further UI in this frame
                if elapsed >= CALIBRATION_TIME:
                    baseline_pitch = np.mean(pitch_samples)
                    baseline_yaw = np.mean(yaw_samples)
                    calibrated = True

                    print("✅ Calibration Complete")
                    print("Baseline Pitch:", baseline_pitch)
                    print("Baseline Yaw:", baseline_yaw)

                cv2.imshow("SafeDrive-AI | FaceMesh + Head Pose", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue   # ⬅️ THIS IS THE KEY LINE



    cv2.imshow("SafeDrive-AI | FaceMesh + Head Pose", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

