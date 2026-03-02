
# SafeDrive AI
**Real-Time Motor Insurance Risk Mitigation System**  


```markdown
#  SafeDrive-AI  
### Personalized, On-Device Driver Safety & Fatigue Intelligence

SafeDrive-AI is a **real-time, edge-based driver monitoring system** that goes beyond basic drowsiness detection.  
It combines **facial landmarks, head-pose analysis, personalized calibration, and fatigue prediction** to detect **dangerous driving behavior before accidents happen**.

>  Focus: **Pre-Accident Intelligence**, ! post-accident detection.

---

##  Key Features (What Makes This Project Unique)

### 1. Head Pose–Based Distraction Detection
- Estimates **Pitch, Yaw, Roll** in real time
- Differentiates between:
  - Drowsiness
  -  Phone usage (looking down)
  -  Side distraction

---

###  2. Personalized Driver Calibration
- First 3 seconds used to learn **individual baseline posture**
- Alerts are based on **relative deviation**, not fixed thresholds
- Reduces false positives across different drivers

---

###  3. Predictive Fatigue (Yawn Engine)
- Uses **Mouth Aspect Ratio (MAR)** to detect yawns
- Tracks **yawn frequency**
- Triggers **fatigue warning before eye closure**

> This predicts fatigue *before* a driver falls asleep.

---

###  4. Edge-First & Privacy-Preserving
- Runs **fully on-device**
- No cloud video upload
- Camera frames never leave the system

---

##  System Architecture

```

Camera Feed
↓
MediaPipe Face Mesh (468 landmarks)
↓
Head Pose Estimation (Pitch / Yaw / Roll)
↓
Personalized Calibration
↓
Fatigue & Distraction Intelligence
↓
Real-Time Alerts & Visual Feedback

````

---

Tech Stack used


| Component | Technology |
|--------|------------|
| Vision & Landmarks | MediaPipe Tasks |
| Computer Vision | OpenCV |
| Math / ML Logic | NumPy |
| Runtime | Python 3.11 |
| Platform | macOS / Linux (Edge-Ready) |

---

##  How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Sandeeprdy1729/SafeDrive-Ai.git
cd SafeDrive-Ai
````

---

### 2️⃣ Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Download MediaPipe Face Model

```bash
curl -L -o face_landmarker.task \
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

---

### 5️⃣ Run the Application

```bash
python app.py
```

---

## 🎥 Usage Instructions

1. Sit normally in front of the camera
2. Look straight for **3 seconds** (calibration phase)
3. Start driving simulation:

   * Look down → 📱 Phone warning
   * Look sideways → 👀 Distraction warning
   * Yawn repeatedly → ⚠️ Fatigue alert

Press **`q`** to quit.

---

## 📊 Demo-Ready Talking Points (For Judges)

* “We don’t wait for accidents — we predict risky behavior early.”
* “Calibration makes the system driver-specific.”
* “Yawns are a fatigue signal before drowsiness.”
* “Runs entirely on-device, preserving privacy.”

---

## 🔮 Future Enhancements

* Near-Miss Black Box Logger
* Emergency SMS & Torch Alerts
* Session Summary Dashboard
* Mobile App (Flutter + TFLite)

---

## 👨‍💻 Author

**Sandeep Reddy Thummala**
GitHub: [https://github.com/Sandeeprdy1729](https://github.com/Sandeeprdy1729)

````

---

# 📦 **requirements.txt (CLEAN & CORRECT)**

Create a file called **`requirements.txt`** and paste this:

```txt
mediapipe==0.10.31
opencv-python==4.12.0.88
numpy>=1.23
````

✅ No OS-specific binaries
✅ No virtual environment files
✅ Works on fresh clone

---

## 🧪 OPTIONAL (Verify clean install)

If you want to double-check:

```bash
pip uninstall -y mediapipe opencv-python numpy
pip install -r requirements.txt
```

Then run:

```bash
python app.py
```

---
Safe drive is to save the accidents caused due to driver distraction

