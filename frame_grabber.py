import cv2
import numpy as np
import threading
import time

class FrameGrabber:
    """Continiously gets the latest frame from the drone video stream."""
    def __init__(self, tello):
        self.tello = tello
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.frame_read = None
        self.last_frame_time = 0.0
        self.last_restart_time = 0.0
        self.last_error_log_time = 0.0
        self.last_stall_log_time = 0.0

    def start(self):
        self.running = True
        thread = threading.Thread(target=self._update, daemon=True)
        thread.start()
        return self

    def _init_frame_reader(self):
        self.frame_read = self.tello.get_frame_read()
        self.last_frame_time = time.time()

    def _restart_stream(self):
        now = time.time()
        if now - self.last_frame_time <= 3.0:
            return

        if self._is_airborne():
            if now - self.last_stall_log_time >= 5.0:
                print("[WARNING] Video stream stalled during flight. Delaying stream restart until landed.")
                self.last_stall_log_time = now
            return

        if now - self.last_restart_time < 5.0:
            return

        self.last_restart_time = now
        print("[WARNING] Video stream stalled. Restarting stream reader...")

        try:
            if self.frame_read is not None:
                self.frame_read.stop()
        except Exception:
            pass

        try:
            self.tello.background_frame_read = None
        except Exception:
            pass

        try:
            self.tello.streamoff()
            time.sleep(0.5)
        except Exception:
            pass

        try:
            self.tello.streamon()
            time.sleep(1.0)
            self._init_frame_reader()
            print("[INFO] Video stream reader restarted.")
        except Exception as exc:
            self.frame_read = None
            self._log_stream_warning(f"[WARNING] Video stream restart failed: {exc}")

    def _is_airborne(self):
        try:
            return self.tello.get_height() > 10
        except Exception:
            return False

    def _log_stream_warning(self, message):
        now = time.time()
        if now - self.last_error_log_time >= 2.0:
            print(message)
            self.last_error_log_time = now

    def _handle_stream_failure(self, exc=None):
        if exc is not None:
            self._log_stream_warning(f"[WARNING] Video frame read failed: {exc}")

        if time.time() - self.last_frame_time > 3.0:
            self._restart_stream()

    def _update(self):
        try:
            
            self._init_frame_reader()
        except Exception as e:
            print(f"[ERROR] Failed to initialize video reader: {e}")
            return

        while self.running:
            try:
                
                frame = getattr(self.frame_read, "frame", None) if self.frame_read is not None else None

                if frame is None or getattr(frame, "size", 0) == 0:
                    self._handle_stream_failure()
                    time.sleep(0.01)
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255).astype(np.uint8)
                frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)

                with self.lock:
                    self.frame = frame

                self.last_frame_time = time.time()
            except Exception as exc:
                self._handle_stream_failure(exc)

            time.sleep(0.01)

    def get_frame(self):
        """Returns the most recent frame instantly (NO QUEUE)."""
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None

    def stop(self):
        self.running = False
        try:
            if self.frame_read is not None:
                self.frame_read.stop()
        except Exception:
            pass
