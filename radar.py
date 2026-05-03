import time


class RadarSystem:
    """Advanced radar system with angle, distance, persistence, and smoothing."""

    def __init__(
        self,
        max_points=100,
        decay_time=3.0,
        angle_merge_threshold=8.0,
        distance_merge_threshold=40.0,
        smoothing=0.35,
    ):
        self.points = []
        self.max_points = max_points
        self.decay_time = decay_time
        self.angle_merge_threshold = angle_merge_threshold
        self.distance_merge_threshold = distance_merge_threshold
        self.smoothing = smoothing
        self.latest_updates = []

    def reset(self):
        self.points = []
        self.latest_updates = []

    def _angle_delta(self, angle_a, angle_b):
        return abs((angle_a - angle_b + 180.0) % 360.0 - 180.0)

    def _find_match(self, label, angle, distance, current_time):
        best_point = None
        best_score = None

        for point in self.points:
            if point["label"] != label:
                continue
            if (current_time - point["timestamp"]) > self.decay_time:
                continue

            angle_delta = self._angle_delta(point["angle"], angle)
            distance_delta = abs(point["distance"] - distance)
            if angle_delta > self.angle_merge_threshold:
                continue
            if distance_delta > self.distance_merge_threshold:
                continue

            score = angle_delta + (distance_delta * 0.1)
            if best_score is None or score < best_score:
                best_score = score
                best_point = point

        return best_point

    def update(self, detections, frame_width):
        """Update radar points from YOLO detections."""
        current_time = time.time()
        self.latest_updates = []
        self.decay(current_time)

        for detection in detections:
            bbox = detection["bbox"]
            conf = detection["conf"]
            label = detection["class"]

            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            area = (x2 - x1) * (y2 - y1)

        
            angle = (cx - frame_width / 2) / frame_width * 90

            
            distance = max(20, min(300, 50000 / (area + 1)))

            matched_point = self._find_match(label, angle, distance, current_time)
            if matched_point is None:
                point = {
                    "angle": angle,
                    "distance": distance,
                    "label": label,
                    "confidence": conf,
                    "timestamp": current_time,
                }
                self.points.append(point)
                self.latest_updates.append(point.copy())
                continue

            alpha = self.smoothing
            matched_point["angle"] = ((1.0 - alpha) * matched_point["angle"]) + (alpha * angle)
            matched_point["distance"] = ((1.0 - alpha) * matched_point["distance"]) + (alpha * distance)
            matched_point["confidence"] = max(matched_point["confidence"], conf)
            matched_point["timestamp"] = current_time
            self.latest_updates.append(matched_point.copy())

        if len(self.points) > self.max_points:
            self.points = self.points[-self.max_points:]

        return [point.copy() for point in self.latest_updates]

    def decay(self, current_time=None):
        """Remove old points based on decay time."""
        current_time = time.time() if current_time is None else current_time
        self.points = [
            point for point in self.points
            if (current_time - point["timestamp"]) <= self.decay_time
        ]

    def get_points(self):
        """Return current radar points."""
        return [point.copy() for point in self.points]
