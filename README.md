# ResQ-Link
# 🚁 ResQ-Link: Autonomous Drone-Based Search & Rescue System

ResQ-Link is a **real-time AI-powered drone system** designed for search-and-rescue operations.
It detects **victims and fire hazards**, navigates autonomously, and makes intelligent decisions using computer vision and control logic.

---

## 🔥 Core Capabilities

### 🎯 Multi-Object Detection

* 👤 **Victim Detection** (YOLOv8 – human detection)
* 🔥 **Fire Detection**

  * Supports **small flames (matchstick/lighter)**
  * Uses **AI + HSV validation** to reduce false positives

---

### 🧠 Autonomous Intelligence

#### 🔄 Rotation Scan Mode

* Smooth **continuous 360° rotation**
* No step-based jitter (RC control based)

#### 🎯 Victim Tracking

* Centers target in frame
* Maintains optimal distance

#### 🔥 Fire Avoidance System

* Detects nearby fire
* Executes intelligent escape:

```text
STOP → ROTATE 180° → MOVE AWAY
```

---

### ⚡ Real-Time Architecture (NO LAG)

```text
Drone Camera
     ↓
Frame Grabber Thread (latest frame only)
     ↓
Main Loop (Display + Control)
     ↓
Async Detection Thread (YOLO)
     ↓
Shared Detections
     ↓
Autonomous Controller
     ↓
Drone Commands
```

✔ Non-blocking
✔ Parallel processing
✔ Stable FPS

---

## 🧪 Detection Pipeline

```text
YOLO Detection
     ↓
Area Filtering
     ↓
HSV Color Validation
     ↓
Brightness Check
     ↓
Adaptive Thresholding
```

✔ Reduces false positives
✔ Detects small flames
✔ Works in real-time

---

## 🎮 Controls

| Key       | Action               |
| --------- | -------------------- |
| `t`       | Takeoff              |
| `l`       | Land                 |
| `y`       | Toggle Rotation Scan |
| `u`       | Demo Script          |
| `w/a/s/d` | Manual Movement      |
| `x`       | Hover                |
| `q`       | Quit                 |

---

## ⚙️ Tech Stack

* Python
* OpenCV
* Ultralytics YOLOv8
* djitellopy (Tello SDK)
* Multi-threading

---

## 🚀 Setup & Run

```bash
pip install ultralytics opencv-python djitellopy numpy
```

```bash
python main.py
```

---

## ⚙️ Configuration

Modify `config.py`:

* Detection thresholds
* Fire sensitivity
* Drone speeds
* Rotation speed
* UI parameters

---

## 🧠 Key Engineering Highlights

### ✅ Async Detection Engine

* Runs YOLO in parallel thread
* No impact on video stream

### ✅ Hybrid Fire Detection

* Deep learning + rule-based filtering
* Handles:

  * matchstick 🔥
  * lighter 🔥
  * candle 🔥

### ✅ Smooth Rotation System

* Uses `send_rc_control` instead of step commands
* Eliminates jerky motion

### ✅ State-Based Control Logic

* Modes:

  * MANUAL
  * ROTATION_SCAN
  * DEMO_SCRIPT

---

## ⚠️ Limitations

* Sensitive to lighting conditions
* HSV thresholds may need tuning per environment
* Small flame detection depends on camera quality

---

## 🔮 Future Improvements

* 🔊 Audio alert system
* 🧭 Path memory + mapping
* 🤖 Multi-drone coordination
* ☁️ Cloud monitoring dashboard

---

## 🏁 Conclusion

ResQ-Link combines:

* AI perception
* Real-time control
* Autonomous decision-making

to create a **practical rescue drone system**, not just a detection demo.

---

## 👨‍💻 Author

**Tarun Singh Thakur**
Engineering Student | AI Systems Builder

---

## ⭐ Support

If you found this useful, consider starring ⭐ the repo.
