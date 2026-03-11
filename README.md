# 🖱️ HeadMouse — Hands-Free Cursor Control via Head Tracking

> A hybrid assistive technology system that enables people with upper-limb disabilities to control their computer cursor using only head movements — while keeping the physical mouse exclusively for clicking and scrolling.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![dlib](https://img.shields.io/badge/dlib-19.x-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

---

## 🎯 Concept

Most head-tracking systems try to handle **everything** through the camera — movement, clicking, scrolling — leading to false triggers and frustrating user experiences.

**HeadMouse takes a different approach:**

| Input | Responsibility |
|-------|---------------|
| 📷 Camera | Cursor movement only |
| 🖱️ Physical mouse | Clicks & scroll only (movement locked) |

The physical mouse is **position-locked** via a `pynput` listener. When the user moves it, it snaps back instantly to the camera-computed position. This means the user can rest their hand on the mouse and click/scroll naturally — without accidentally moving the cursor.

---

## ✨ Key Features

- **Relative movement architecture** — cursor moves based on *how much* the head deviates from neutral, not *where* the head is. This means the cursor can reach every corner of the screen with minimal head movement.
- **Automatic calibration** — the system measures each user's neutral nose position over 60 frames at startup. No manual configuration needed.
- **Speed curve** — small head movements → slow precision cursor; large deviations → fast cursor. Configurable.
- **Dead zone** — minor tremors and micro-movements are filtered out, preventing unintended drift.
- **CLAHE lighting correction** — works under variable lighting conditions.
- **Physical mouse lock** — the mouse stays put; only clicks and scroll events pass through.

---

## 🧠 How It Works

```
┌─────────────────────────────────────────────────────┐
│                   CAMERA FEED                       │
│                                                     │
│  dlib face detector → 68 facial landmarks           │
│         ↓                                           │
│  Nose tip (landmark #30) tracked                    │
│         ↓                                           │
│  Deviation from calibrated neutral point            │
│         ↓                                           │
│  Speed curve applied  →  velocity (vx, vy)          │
│         ↓                                           │
│  LightSmooth filter (tremor suppression)            │
│         ↓                                           │
│  pyautogui.moveTo(cursor_x, cursor_y)               │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│              PHYSICAL MOUSE (pynput)                │
│                                                     │
│  on_move() → snap back to camera target             │
│  on_click() → pass through normally ✓               │
│  on_scroll() → pass through normally ✓              │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Requirements

- Python 3.8+
- Webcam
- `shape_predictor_68_face_landmarks.dat` (dlib model file)

### Install dependencies

```bash
pip install opencv-python dlib imutils pyautogui pynput numpy
```

### Download dlib landmark model

```bash
# Download from dlib's official source
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

### Run

```bash
python hibrit_fare_v2.py
```

---

## ⚙️ Configuration

All tunable parameters are at the top of the script:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `DEAD_ZONE` | `8` | Minimum pixel deviation to register movement. Lower = more sensitive but may drift. |
| `SPEED` | `8.0` | Base speed multiplier. Higher = faster cursor. |
| `SPEED_CURVE` | `1.4` | Exponential curve. `1.0` = linear, higher values make large movements proportionally faster. |
| `LightSmooth alpha` | `0.5` | Smoothing factor. `1.0` = raw, `0.3` = heavy smoothing. |

**Tuning tip:** Start with `SPEED` only. Touch `DEAD_ZONE` only if you experience drift or if the cursor feels unresponsive.

---

## 🏥 Target Users

This project is designed for individuals with **upper-limb motor impairments** who:
- Can move their head freely
- Have limited or no hand/arm mobility
- Want to use standard software without specialized hardware

---

## 📚 Related Work

This project builds on and differentiates from existing head-tracking research:

- **BlinkMouse** (IJSAT 2025) — dlib-based landmark tracking with blink-based clicking
- **EMKEY** — head movement + voice keyboard emulation
- **Head + Sip-and-Puff Mouse** (MDPI 2025) — pneumatic switch for clicking

**Key differentiator:** HeadMouse offloads all click/scroll input to the physical mouse, eliminating false-positive triggers — the primary failure mode of camera-only systems.

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

## 🤝 Contributing

Issues and pull requests are welcome. Areas for improvement:
- Multi-monitor support
- GUI settings panel
- MacOS / Linux testing
- Accuracy benchmarking
