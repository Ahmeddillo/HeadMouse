import cv2
import dlib
import numpy as np
from imutils import face_utils
import pyautogui
import time
from collections import deque
from pynput import mouse as pynput_mouse
import threading

print("🚀 HeadMouse v2 - Relative Motion")
print("📌 Camera: Head movement → direction | Physical mouse: Click & Scroll")
print("📌 Press 'q' to exit")

# ------
#  SCREEN & CAMERA
# ------
screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# ------
#  PHYSICAL MOUSE LOCK
# ------
target_x = float(screen_w // 2)
target_y = float(screen_h // 2)
state_lock = threading.Lock()

def on_move(x, y):
    """Sync physical mouse position with camera target."""
    with state_lock:
        tx, ty = target_x, target_y
    if abs(x - tx) > 4 or abs(y - ty) > 4:
        pyautogui.moveTo(int(tx), int(ty))

mouse_listener = pynput_mouse.Listener(on_move=on_move)
mouse_listener.start()

# ------
#  RELATIVE MOTION PARAMETERS
#
#  Idea: Not "where" the nose is, but how much it 
#  deviates from the center point.
#
#  Deviation -> Velocity -> Cursor moves in that direction.
#  Higher deviation = Faster movement.
#  Back to center = Cursor stops.
# ------

# Center of the nose measured during calibration
origin_x = cam_w // 2
origin_y = cam_h // 2

# Dead zone: Ignore small movements (anti-jitter)
DEAD_ZONE = 8          # pixels

# Speed multiplier: deviation * this factor = pixels/frame
SPEED = 1.0           

# Velocity Curve: Speed increases exponentially as deviation grows
# 1.0 = Linear, 1.5 = Exponential
SPEED_CURVE = 1.4

# ------
#  CALIBRATION
#  Measures the neutral position while the user looks straight.
# ------
class Calibrator:
    def __init__(self, n=60):
        self.n        = n
        self.nose_x   = []
        self.nose_y   = []
        self.done     = False
        self.origin_x = cam_w // 2
        self.origin_y = cam_h // 2

    def update(self, nx, ny):
        if self.done:
            return
        self.nose_x.append(nx)
        self.nose_y.append(ny)
        if len(self.nose_x) >= self.n:
            self.origin_x = int(np.mean(self.nose_x))
            self.origin_y = int(np.mean(self.nose_y))
            self.done = True
            print(f"✅ Neutral Position Set: ({self.origin_x}, {self.origin_y})")

    @property
    def progress(self):
        return min(len(self.nose_x) / self.n, 1.0)

# ------
#  LIGHT SMOOTHING
#  Reduces frame-to-frame noise without high latency.
# ------
class LightSmooth:
    def __init__(self, alpha=0.5):
        # alpha: 1.0 = raw, 0.3 = very smooth
        self.alpha = alpha
        self.val   = None

    def update(self, x):
        if self.val is None:
            self.val = x
        self.val = self.alpha * x + (1 - self.alpha) * self.val
        return self.val

# ------
#  OBJECT INITIALIZATION
# ------
calib    = Calibrator(n=60)
smooth_x = LightSmooth(alpha=0.5)
smooth_y = LightSmooth(alpha=0.5)
fps_buf  = deque(maxlen=30)

# ------
#  HELPER FUNCTIONS
# ------
def improve_light(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def calc_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C) if C > 1e-6 else 0.3

def apply_speed_curve(delta, curve, speed, dead_zone):
    if abs(delta) < dead_zone:
        return 0.0
    sign = 1 if delta > 0 else -1
    effective = abs(delta) - dead_zone
    velocity = speed * (effective ** curve) / (dead_zone ** (curve - 1))
    return sign * velocity

def draw_crosshair(frame, cx, cy, size=12, color=(0, 255, 80)):
    """Display the neutral/origin position."""
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, 1)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, 1)
    cv2.circle(frame, (cx, cy), 3, color, -1)

def draw_face_rect(frame, rect, color):
    P, L = 20, 14
    x1 = max(0, rect.left()  - P)
    y1 = max(0, rect.top()   - P)
    x2 = min(cam_w, rect.right()  + P)
    y2 = min(cam_h, rect.bottom() + P)
    for (ax,ay),(bx,by),(cx,cy) in [
        ((x1,y1),(x1+L,y1),(x1,y1+L)),
        ((x2,y1),(x2-L,y1),(x2,y1+L)),
        ((x1,y2),(x1+L,y2),(x1,y2-L)),
        ((x2,y2),(x2-L,y2),(x2,y2-L)),
    ]:
        cv2.line(frame, (ax,ay),(bx,by), color, 2)
        cv2.line(frame, (ax,ay),(cx,cy), color, 2)

def draw_velocity_arrow(frame, cx, cy, vx, vy):
    """Show the direction and speed of movement."""
    scale = 0.8
    ex = int(cx + vx * scale)
    ey = int(cy + vy * scale)
    if abs(vx) > 1 or abs(vy) > 1:
        cv2.arrowedLine(frame, (cx, cy), (ex, ey), (0, 200, 255), 2, tipLength=0.3)

def draw_ui(frame, fps, calib_done, vx, vy, cur_x, cur_y):
    cv2.rectangle(frame, (0, 0), (380, 82), (15, 15, 15), -1)
    cv2.putText(frame, f"FPS: {fps:.0f}  |  EXIT: 'q'",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1)
    if calib_done:
        cv2.putText(frame, "Center: Stop | Tilt: Move Cursor",
                    (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 120), 1)
        cv2.putText(frame, f"Vel: ({vx:+.0f}, {vy:+.0f})  Cursor: ({cur_x}, {cur_y})",
                    (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 0), 1)
        cv2.putText(frame, "PHYSICAL MOUSE: Click & Scroll",
                    (8, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
    else:
        cv2.putText(frame, "CALIBRATION: Look straight, stay still",
                    (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 180, 255), 1)

def draw_progress(frame, progress):
    bw = cam_w - 40
    cv2.rectangle(frame, (20, cam_h-38), (20+bw, cam_h-18), (50,50,50), -1)
    cv2.rectangle(frame, (20, cam_h-38), (20+int(bw*progress), cam_h-18), (0,210,100), -1)
    cv2.putText(frame, f"Calibration %{int(progress*100)} — Look straight",
                (20, cam_h-43), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180,180,180), 1)

# ------
#  MAIN LOOP
# ------
cur_x, cur_y = float(screen_w // 2), float(screen_h // 2)
vx_disp, vy_disp = 0.0, 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t_now = time.time()
    fps_buf.append(t_now)
    fps = (len(fps_buf) / (fps_buf[-1] - fps_buf[0] + 1e-6)) if len(fps_buf) > 1 else 0

    frame = cv2.flip(frame, 1)
    frame = improve_light(frame)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        draw_face_rect(frame, rect,
                       color=(0,220,80) if calib.done else (0,160,255))

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        nose_x, nose_y = int(shape[30][0]), int(shape[30][1])

        # ------ Calibration ------
        if not calib.done:
            calib.update(nose_x, nose_y)
            draw_progress(frame, calib.progress)
            cv2.circle(frame, (nose_x, nose_y), 6, (0,255,0), -1)
            draw_crosshair(frame, cam_w//2, cam_h//2, color=(100,100,100))
            break

        # ------ Show Neutral Origin ------
        ox, oy = calib.origin_x, calib.origin_y
        draw_crosshair(frame, ox, oy)

        # ------ Smoothed Nose Position ------
        sx = smooth_x.update(nose_x)
        sy = smooth_y.update(nose_y)
        cv2.circle(frame, (int(sx), int(sy)), 6, (0,255,80), -1)

        # ------ Deviation from Center ------
        delta_x = sx - ox
        delta_y = sy - oy

        # ------ Deviation -> Velocity ------
        vx = apply_speed_curve(delta_x, SPEED_CURVE, SPEED, DEAD_ZONE)
        vy = apply_speed_curve(delta_y, SPEED_CURVE, SPEED, DEAD_ZONE)
        vx_disp, vy_disp = vx, vy

        # Draw Velocity Arrow
        draw_velocity_arrow(frame, int(sx), int(sy), vx * 2, vy * 2)

        # ------ Update Cursor Position ------
        cur_x = np.clip(cur_x + vx, 0, screen_w - 1)
        cur_y = np.clip(cur_y + vy, 0, screen_h - 1)

        with state_lock:
            target_x = cur_x
            target_y = cur_y

        try:
            pyautogui.moveTo(int(cur_x), int(cur_y))
        except Exception:
            pass

        break  # Single face only

    draw_ui(frame, fps, calib.done, vx_disp, vy_disp, int(cur_x), int(cur_y))
    if not calib.done:
        draw_progress(frame, calib.progress)

    cv2.imshow("Hybrid Mouse v2 - Relative Motion", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------
#  CLEANUP
# ------
mouse_listener.stop()
cap.release()
cv2.destroyAllWindows()
print("👋 Control returned to you!")
