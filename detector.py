import threading
import time
import numpy as np

import cv2
from ultralytics import YOLO

from config import Config
from utils import get_area


class Detector:
    """Asynchronous YOLO detector running in a background thread."""

    def __init__(self):
        self.person_model = YOLO(Config.PERSON_MODEL)
        self.fire_model = YOLO(Config.FIRE_MODEL)

        self.latest_frame = None
        self.latest_detections = []
        self.latest_fire_detections = []
        self.latest_person_detections = []
        self.detection_history = []
        self.inference_fps = 0.0
        self.run_fire_next = True

        self.running = False
        self.lock = threading.Lock()
        self.frame_event = threading.Event()

    def start(self):
        self.running = True
        thread = threading.Thread(target=self._detect_loop, daemon=True)
        thread.start()
        return self

    def update_frame(self, frame):
        """Provide the newest frame to the detector."""
        with self.lock:
            self.latest_frame = frame.copy()
        self.frame_event.set()

    def _scale_bbox(self, xyxy, scale_x, scale_y):
        x1, y1, x2, y2 = xyxy
        return [
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y),
        ]

    def _detect_loop(self):
        last_time = time.time()

        while self.running:
            if not self.frame_event.wait(timeout=0.1):
                continue

            self.frame_event.clear()

            with self.lock:
                if self.latest_frame is None:
                    continue

                source_frame = self.latest_frame

            process_width = min(Config.YOLO_WIDTH, 256)
            process_height = int(process_width * Config.CAMERA_HEIGHT / Config.CAMERA_WIDTH)
            frame_to_process = cv2.resize(source_frame, (process_width, process_height))

            scale_x = Config.CAMERA_WIDTH / process_width
            scale_y = Config.CAMERA_HEIGHT / process_height
            run_fire_model = self.run_fire_next
            self.run_fire_next = not self.run_fire_next
            fire_detections = None
            person_detections = None

            if run_fire_model:
                fire_detections = []
                fire_results = self.fire_model.predict(
                    source=frame_to_process,
                    conf=Config.FIRE_CONFIDENCE_THRESHOLD,
                    imgsz=process_width,
                    verbose=False,
                )

                for result in fire_results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        bbox = self._scale_bbox([x1, y1, x2, y2], scale_x, scale_y)

                        area = get_area(bbox)

                        if area < 800:
                            continue

                        roi = frame_to_process[int(y1):int(y2), int(x1):int(x2)]

                        if roi.size == 0:
                            continue

                        roi = cv2.GaussianBlur(roi, (3, 3), 0)
                        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                        lower1 = np.array([5, 80, 80])
                        upper1 = np.array([35, 255, 255])

                        lower2 = np.array([0, 120, 120])
                        upper2 = np.array([10, 255, 255])

                        mask1 = cv2.inRange(hsv, lower1, upper1)
                        mask2 = cv2.inRange(hsv, lower2, upper2)
                        mask = cv2.bitwise_or(mask1, mask2)

                        fire_ratio = cv2.countNonZero(mask) / (roi.shape[0] * roi.shape[1])

                        if area < 5000:
                            if fire_ratio < 0.08:
                                continue
                        else:
                            if fire_ratio < 0.2:
                                continue

                        brightness = np.mean(hsv[:, :, 2])

                        if area < 5000:
                            if brightness < 70:
                                continue
                        else:
                            if brightness < 110:
                                continue

                        w = x2 - x1
                        h = y2 - y1

                        if w > h * 2:
                            continue

                        fire_detections.append(
                            {
                                "bbox": bbox,
                                "conf": float(box.conf[0].item()),
                                "class": "fire",
                            }
                        )

            else:
                person_detections = []
                person_results = self.person_model.predict(
                    source=frame_to_process,
                    conf=Config.CONFIDENCE_THRESHOLD,
                    imgsz=process_width,
                    verbose=False,
                )

                for result in person_results:
                    for box in result.boxes:
                        cls_id = int(box.cls[0].item())
                        if cls_id != 0:
                            continue

                        person_detections.append(
                            {
                                "bbox": self._scale_bbox(box.xyxy[0].tolist(), scale_x, scale_y),
                                "conf": float(box.conf[0].item()),
                                "class": "victim",
                            }
                        )

            if fire_detections is not None:
                self.latest_fire_detections = fire_detections
            if person_detections is not None:
                self.latest_person_detections = person_detections

            detections = list(self.latest_fire_detections) + list(self.latest_person_detections)

            self.detection_history.append(detections)
            if len(self.detection_history) > 3:
                self.detection_history.pop(0)

            smoothed_detections = detections
            if not smoothed_detections:
                for past_dets in reversed(self.detection_history[:-1]):
                    if past_dets:
                        smoothed_detections = past_dets
                        break

            curr_time = time.time()
            fps = 1.0 / (curr_time - last_time + 1e-5)
            last_time = curr_time

            with self.lock:
                self.latest_detections = smoothed_detections
                self.inference_fps = fps

    def get_detections(self):
        """Return the newest detections and inference FPS."""
        with self.lock:
            return self.latest_detections, self.inference_fps

    def stop(self):
        self.running = False
