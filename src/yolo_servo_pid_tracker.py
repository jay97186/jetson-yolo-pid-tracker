import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

import numpy as np
import cv2
import torch
from ultralytics import YOLO

import board
import busio
from adafruit_servokit import ServoKit
from time import time
from collections import deque


# ─────────────────────────────────────────────
#  PID Controller
# ─────────────────────────────────────────────
class PIDController:
    """Simple PID controller for single-axis servo tracking."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_min: float = -5.0,
        output_max: float = 5.0,
        integral_limit: float = 1000.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral_limit = integral_limit

        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = time()

        # Per-term history for analysis
        self.last_p = 0.0
        self.last_i = 0.0
        self.last_d = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = time()

    def compute(self, error: float) -> float:
        now = time()
        dt = now - self._prev_time
        if dt <= 0:
            dt = 1e-3

        # P
        p = self.kp * error

        # I
        self._integral += error * dt
        self._integral = float(np.clip(self._integral, -self.integral_limit, self.integral_limit))
        i = self.ki * self._integral

        # D
        derivative = (error - self._prev_error) / dt
        d = self.kd * derivative

        self._prev_error = error
        self._prev_time = now

        self.last_p = p
        self.last_i = i
        self.last_d = d

        output = p + i + d
        return float(np.clip(output, self.output_min, self.output_max))


# ─────────────────────────────────────────────
#  PID Response Analyser
# ─────────────────────────────────────────────
class PIDAnalyser:
    """
    Maintains rolling buffers and renders a 4-panel analysis chart
    directly onto an OpenCV image.

    Panels
    ──────
    1. Error (px)          – set-point error over time
    2. P / I / D terms     – individual PID contributions
    3. Correction (deg)    – output applied to the servo
    4. Servo angle (deg)   – absolute tilt position
    """

    HISTORY = 300          # samples kept (~10 s at 30 fps)
    PANEL_H = 120          # height of each chart panel (px)
    PANEL_W = 640          # width of the analysis strip
    PADDING = 8            # inner padding
    LABEL_W = 60           # left label column width

    # BGR colours
    C_ERROR      = (0,   200, 255)   # amber
    C_P          = (255, 100,  50)   # blue
    C_I          = ( 50, 255, 100)   # green
    C_D          = ( 50, 100, 255)   # red
    C_CORRECTION = (200, 200,   0)   # cyan
    C_ANGLE      = (200,  50, 200)   # purple
    C_ZERO       = ( 80,  80,  80)   # grey zero-line
    C_DEAD       = ( 40, 120,  40)   # dark-green dead-zone band
    C_BG         = ( 18,  18,  18)   # near-black background
    C_GRID       = ( 35,  35,  35)   # subtle grid lines
    C_TEXT       = (200, 200, 200)   # label text

    def __init__(self, dead_zone: float, servo_soft_min: float, servo_soft_max: float):
        self.dead_zone = dead_zone
        self.servo_soft_min = servo_soft_min
        self.servo_soft_max = servo_soft_max

        self.t_buf         = deque(maxlen=self.HISTORY)
        self.error_buf     = deque(maxlen=self.HISTORY)
        self.p_buf         = deque(maxlen=self.HISTORY)
        self.i_buf         = deque(maxlen=self.HISTORY)
        self.d_buf         = deque(maxlen=self.HISTORY)
        self.corr_buf      = deque(maxlen=self.HISTORY)
        self.angle_buf     = deque(maxlen=self.HISTORY)

        self._t0 = time()

    def push(self, error, p, i, d, correction, angle):
        self.t_buf.append(time() - self._t0)
        self.error_buf.append(error)
        self.p_buf.append(p)
        self.i_buf.append(i)
        self.d_buf.append(d)
        self.corr_buf.append(correction)
        self.angle_buf.append(angle)

    # ── helpers ──────────────────────────────

    def _make_canvas(self) -> np.ndarray:
        total_h = self.PANEL_H * 4 + self.PADDING * 5
        canvas = np.full((total_h, self.PANEL_W, 3), self.C_BG, dtype=np.uint8)
        return canvas

    def _panel_rect(self, idx: int):
        """Return (x0, y0, x1, y1) for panel index 0-3."""
        x0 = self.LABEL_W
        y0 = self.PADDING + idx * (self.PANEL_H + self.PADDING)
        x1 = self.PANEL_W - self.PADDING
        y1 = y0 + self.PANEL_H
        return x0, y0, x1, y1

    def _draw_grid(self, canvas, x0, y0, x1, y1, n_h=4):
        """Faint horizontal grid lines."""
        for k in range(1, n_h):
            gy = y0 + int((y1 - y0) * k / n_h)
            cv2.line(canvas, (x0, gy), (x1, gy), self.C_GRID, 1)

    def _plot_series(self, canvas, x0, y0, x1, y1, values, color,
                     v_min=None, v_max=None, thickness=1):
        """Plot a list of values as a polyline within the panel rect."""
        n = len(values)
        if n < 2:
            return
        arr = np.array(values, dtype=np.float32)
        lo = v_min if v_min is not None else arr.min()
        hi = v_max if v_max is not None else arr.max()
        span = hi - lo
        if span < 1e-6:
            span = 1.0

        pw = x1 - x0
        ph = y1 - y0

        pts = []
        for k, v in enumerate(arr):
            px = x0 + int(pw * k / (n - 1))
            py = y1 - int(ph * (v - lo) / span)
            py = int(np.clip(py, y0, y1))
            pts.append((px, py))

        pts_np = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts_np], False, color, thickness, cv2.LINE_AA)

    def _zero_line(self, canvas, x0, y0, x1, y1, values, v_min, v_max):
        """Draw horizontal zero line scaled to the same range as the series."""
        span = v_max - v_min
        if span < 1e-6:
            return
        zy = y1 - int((y1 - y0) * (0.0 - v_min) / span)
        zy = int(np.clip(zy, y0, y1))
        cv2.line(canvas, (x0, zy), (x1, zy), self.C_ZERO, 1)

    def _dead_band(self, canvas, x0, y0, x1, y1, v_min, v_max):
        """Shade the dead-zone band on error panel."""
        span = v_max - v_min
        if span < 1e-6:
            return
        top    = y1 - int((y1 - y0) * ( self.dead_zone - v_min) / span)
        bottom = y1 - int((y1 - y0) * (-self.dead_zone - v_min) / span)
        top    = int(np.clip(top,    y0, y1))
        bottom = int(np.clip(bottom, y0, y1))
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, top), (x1, bottom), self.C_DEAD, -1)
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

    def _label(self, canvas, x0, y0, text, color):
        cv2.putText(canvas, text, (4, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    def _stat_line(self, canvas, x1, y1, values, color):
        """Print min / max / last in the bottom-right of a panel."""
        if not values:
            return
        arr = np.array(values)
        txt = f"cur={arr[-1]:+.1f}  min={arr.min():.1f}  max={arr.max():.1f}"
        cv2.putText(canvas, txt, (x1 - 310, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)

    # ── public render ─────────────────────────

    def render(self) -> np.ndarray:
        canvas = self._make_canvas()

        # ── Panel 0: Error ──────────────────
        x0, y0, x1, y1 = self._panel_rect(0)
        self._draw_grid(canvas, x0, y0, x1, y1)
        if self.error_buf:
            arr = np.array(self.error_buf)
            lo = min(arr.min(), -self.dead_zone * 1.5)
            hi = max(arr.max(),  self.dead_zone * 1.5)
            self._dead_band(canvas, x0, y0, x1, y1, lo, hi)
            self._zero_line(canvas, x0, y0, x1, y1, arr, lo, hi)
            self._plot_series(canvas, x0, y0, x1, y1, arr, self.C_ERROR,
                              v_min=lo, v_max=hi, thickness=2)
            self._stat_line(canvas, x1, y1, arr, self.C_ERROR)
        self._label(canvas, x0, y0, "Error(px)", self.C_ERROR)

        # ── Panel 1: P / I / D ──────────────
        x0, y0, x1, y1 = self._panel_rect(1)
        self._draw_grid(canvas, x0, y0, x1, y1)
        all_pid = list(self.p_buf) + list(self.i_buf) + list(self.d_buf)
        if all_pid:
            lo = min(all_pid) * 1.1 - 0.01
            hi = max(all_pid) * 1.1 + 0.01
            self._zero_line(canvas, x0, y0, x1, y1, [], lo, hi)
            self._plot_series(canvas, x0, y0, x1, y1, self.p_buf, self.C_P,
                              v_min=lo, v_max=hi)
            self._plot_series(canvas, x0, y0, x1, y1, self.i_buf, self.C_I,
                              v_min=lo, v_max=hi)
            self._plot_series(canvas, x0, y0, x1, y1, self.d_buf, self.C_D,
                              v_min=lo, v_max=hi)
            # Legend
            for k, (lbl, col) in enumerate([("P", self.C_P),
                                             ("I", self.C_I),
                                             ("D", self.C_D)]):
                lx = x1 - 120 + k * 38
                cv2.putText(canvas, lbl, (lx, y0 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        self._label(canvas, x0, y0, "P/I/D", self.C_TEXT)

        # ── Panel 2: Correction ─────────────
        x0, y0, x1, y1 = self._panel_rect(2)
        self._draw_grid(canvas, x0, y0, x1, y1)
        if self.corr_buf:
            arr = np.array(self.corr_buf)
            lo = min(arr.min() * 1.1, -1.0)
            hi = max(arr.max() * 1.1,  1.0)
            self._zero_line(canvas, x0, y0, x1, y1, arr, lo, hi)
            self._plot_series(canvas, x0, y0, x1, y1, arr, self.C_CORRECTION,
                              v_min=lo, v_max=hi, thickness=2)
            self._stat_line(canvas, x1, y1, arr, self.C_CORRECTION)
        self._label(canvas, x0, y0, "Corr(deg)", self.C_CORRECTION)

        # ── Panel 3: Servo Angle ────────────
        x0, y0, x1, y1 = self._panel_rect(3)
        self._draw_grid(canvas, x0, y0, x1, y1)
        if self.angle_buf:
            arr = np.array(self.angle_buf)
            lo = self.servo_soft_min
            hi = self.servo_soft_max
            # Soft-limit lines
            for lim_v, lim_c in [(lo, (0, 80, 200)), (hi, (0, 80, 200))]:
                ly = y1 - int((y1 - y0) * (lim_v - lo) / (hi - lo))
                ly = int(np.clip(ly, y0, y1))
                cv2.line(canvas, (x0, ly), (x1, ly), lim_c, 1)
            self._plot_series(canvas, x0, y0, x1, y1, arr, self.C_ANGLE,
                              v_min=lo, v_max=hi, thickness=2)
            self._stat_line(canvas, x1, y1, arr, self.C_ANGLE)
        self._label(canvas, x0, y0, "Angle(deg)", self.C_ANGLE)

        # ── Title bar ───────────────────────
        cv2.putText(canvas, "PID Response Analysis",
                    (self.LABEL_W, self.PADDING - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_TEXT, 1, cv2.LINE_AA)

        return canvas


# ─────────────────────────────────────────────
#  GStreamer pipeline
# ─────────────────────────────────────────────
def create_pipeline() -> str:
    return (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink name=sink max-buffers=1 drop=true"
    )


# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
MODEL_PATH = "/home/jetson/yolov8n-face.pt"

SERVO_CHANNEL = 0
SERVO_MIN_PW = 500
SERVO_MAX_PW = 2500
SERVO_INIT = 90
SERVO_SOFT_MIN = 5
SERVO_SOFT_MAX = 175

FRAME_W = 1280
FRAME_CX = FRAME_W / 2

# If servo moves opposite direction, change to True
SERVO_REVERSE = False

# PID gains
KP = 0.003
KI = 0.005
KD = 0.0001

# Max angle correction per frame
PID_OUTPUT_MIN = -30.0
PID_OUTPUT_MAX = 30.0

# Dead zone in pixels:
# when |error| < DEAD_ZONE, servo will not move
DEAD_ZONE = 80

# Show analysis window in a separate window (True) or
# stacked below the camera feed (False)
ANALYSIS_SEPARATE_WINDOW = True


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    # ── GStreamer ──────────────────────────────
    Gst.init(None)
    pipeline = Gst.parse_launch(create_pipeline())
    sink = pipeline.get_by_name("sink")
    if sink is None:
        print("[ERROR] Could not find appsink 'sink' in pipeline.")
        return

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("[ERROR] Failed to start GStreamer pipeline.")
        return
    print("[INFO] GStreamer pipeline started.")

    # ── YOLO ──────────────────────────────────
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    model = YOLO(MODEL_PATH)

    # ── Servo ─────────────────────────────────
    i2c = busio.I2C(board.SCL, board.SDA)
    kit = ServoKit(channels=16, i2c=i2c)
    servo = kit.servo[SERVO_CHANNEL]
    servo.set_pulse_width_range(SERVO_MIN_PW, SERVO_MAX_PW)

    current_angle = float(SERVO_INIT)
    servo.angle = current_angle

    # ── PID ───────────────────────────────────
    pid = PIDController(
        kp=KP,
        ki=KI,
        kd=KD,
        output_min=PID_OUTPUT_MIN,
        output_max=PID_OUTPUT_MAX,
    )

    # ── Analyser ──────────────────────────────
    analyser = PIDAnalyser(
        dead_zone=DEAD_ZONE,
        servo_soft_min=SERVO_SOFT_MIN,
        servo_soft_max=SERVO_SOFT_MAX,
    )

    print("[INFO] Starting inference... Press 'q' or ESC to quit.")

    try:
        while True:
            sample = sink.emit("pull-sample")
            if sample is None:
                continue

            buf = sample.get_buffer()
            caps = sample.get_caps()
            structure = caps.get_structure(0)
            width = structure.get_value("width")
            height = structure.get_value("height")

            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                continue

            try:
                frame = np.frombuffer(map_info.data, np.uint8).reshape((height, width, 3)).copy()
                frame = cv2.flip(frame, 1)  # mirror left-right

                with torch.no_grad():
                    results = model(
                        frame,
                        imgsz=640,
                        conf=0.5,
                        device=device,
                        half=(device != "cpu"),
                        verbose=False,
                    )

                annotated = results[0].plot()
                boxes = results[0].boxes

                error = 0.0
                correction = 0.0

                if boxes is not None and len(boxes) > 0:
                    # pick highest-confidence detection
                    best_idx = int(boxes.conf.argmax())
                    cx_det = float(boxes.xywh[best_idx, 0])

                    # error > 0: target is to the right of image center
                    # error < 0: target is to the left of image center
                    error = cx_det - FRAME_CX

                    # ── Dead zone: small error => don't move servo ──
                    if abs(error) < DEAD_ZONE:
                        correction = 0.0
                        pid._integral *= 0.8  # slowly decay integral to reduce jitter
                    else:
                        correction = pid.compute(error)
                        if SERVO_REVERSE:
                            correction = -correction

                        new_angle = current_angle + correction
                        new_angle = float(np.clip(new_angle, SERVO_SOFT_MIN, SERVO_SOFT_MAX))

                        # Anti-windup when hitting limits
                        if new_angle <= SERVO_SOFT_MIN or new_angle >= SERVO_SOFT_MAX:
                            pid._integral = 0.0

                        current_angle = new_angle
                        servo.angle = current_angle

                    # HUD
                    cv2.circle(annotated, (int(cx_det), height // 2), 8, (0, 255, 0), -1)
                    cv2.line(annotated, (int(FRAME_CX), 0), (int(FRAME_CX), height), (255, 0, 0), 1)
                    cv2.putText(
                        annotated,
                        f"Servo: {current_angle:.1f} deg  err: {error:.1f}px  dead:{DEAD_ZONE}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )
                    if abs(error) < DEAD_ZONE:
                        cv2.putText(
                            annotated, "Dead zone: hold position", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                        )
                    else:
                        cv2.putText(
                            annotated, f"Correction: {correction:.2f}", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                        )

                else:
                    # No detection — decay integral slowly
                    pid._integral *= 0.9
                    cv2.line(annotated, (int(FRAME_CX), 0), (int(FRAME_CX), height), (255, 0, 0), 1)
                    cv2.putText(
                        annotated, "No detection", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                    )

                # ── Feed analyser ──────────────────
                analyser.push(
                    error=error,
                    p=pid.last_p,
                    i=pid.last_i,
                    d=pid.last_d,
                    correction=correction,
                    angle=current_angle,
                )

                # ── Render analysis ────────────────
                analysis_frame = analyser.render()

                if ANALYSIS_SEPARATE_WINDOW:
                    cv2.imshow("YOLOv8 X-Axis PID Servo Tracker", annotated)
                    cv2.imshow("PID Response Analysis", analysis_frame)
                else:
                    # Resize analysis strip to match camera width, stack below
                    af_resized = cv2.resize(
                        analysis_frame,
                        (annotated.shape[1], analysis_frame.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    combined = np.vstack([annotated, af_resized])
                    cv2.imshow("YOLOv8 X-Axis PID Servo Tracker", combined)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

            finally:
                buf.unmap(map_info)

    finally:
        pipeline.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()
        servo.angle = SERVO_INIT
        print("[INFO] Pipeline stopped, servo centred, exiting.")


if __name__ == "__main__":
    main()
