import math
import queue
import threading

from config import Config
from utils import get_area, get_center

try:
    import pyttsx3
except ImportError:  # pragma: no cover - optional dependency
    pyttsx3 = None


class VoiceNarrator:
    """Threaded pyttsx3 wrapper so narration never blocks the main loop."""

    def __init__(self, enabled=True):
        self.enabled = enabled and pyttsx3 is not None
        self._queue = queue.Queue()
        self._running = False
        self._thread = None

    def start(self):
        if not self.enabled or self._running:
            return self

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", Config.VOICE_RATE)
        except Exception as exc:  # pragma: no cover - runtime device issue
            print(f"[WARNING] Voice narration disabled: {exc}")
            self.enabled = False
            self._running = False
            return

        while self._running:
            text = self._queue.get()
            if text is None:
                break

            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:  # pragma: no cover - runtime device issue
                print(f"[WARNING] Voice narration failed: {exc}")

        try:
            engine.stop()
        except Exception:
            pass

    def announce_count(self, count):
        if not self.enabled or not self._running:
            return

        noun = "victim" if count == 1 else "victims"
        self._queue.put(f"{count} {noun} detected")

    def stop(self):
        if not self._running:
            return

        self._running = False
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=1.0)


class MetricsTracker:
    """Passive mission metrics with lightweight victim de-duplication."""

    def __init__(self, narrator=None):
        self.lock = threading.Lock()
        self.narrator = narrator or VoiceNarrator(enabled=Config.VOICE_ENABLED).start()
        self.reset()

    def reset(self):
        with self.lock:
            self.victim_records = []
            self.victim_count = 0
            self.fire_detected = False
            self.last_announced_count = 0

    def _match_record(self, center, area, used_indices):
        best_index = None
        best_distance = float("inf")

        for index, record in enumerate(self.victim_records):
            if index in used_indices:
                continue

            stored_area = max(record["area"], 1)
            area_ratio = min(area, stored_area) / max(area, stored_area)
            if area_ratio < Config.VICTIM_MIN_AREA_RATIO:
                continue

            distance = math.hypot(center[0] - record["center"][0], center[1] - record["center"][1])
            if distance <= Config.VICTIM_MATCH_DISTANCE and distance < best_distance:
                best_distance = distance
                best_index = index

        return best_index

    def update(self, detections):
        victim_detections = [d for d in detections if d.get("class") == "victim"]
        fire_present = any(d.get("class") == "fire" for d in detections)

        announce_count = None
        with self.lock:
            if fire_present:
                self.fire_detected = True

            used_indices = set()
            for detection in sorted(victim_detections, key=lambda item: item["conf"], reverse=True):
                center = get_center(detection["bbox"])
                area = get_area(detection["bbox"])
                match_index = self._match_record(center, area, used_indices)

                if match_index is None:
                    self.victim_records.append({"center": center, "area": area})
                    used_indices.add(len(self.victim_records) - 1)
                    self.victim_count += 1
                else:
                    self.victim_records[match_index]["center"] = center
                    self.victim_records[match_index]["area"] = area
                    used_indices.add(match_index)

            if self.victim_count > self.last_announced_count:
                announce_count = self.victim_count
                self.last_announced_count = self.victim_count

            snapshot = {
                "victim_count": self.victim_count,
                "fire_detected": self.fire_detected,
            }

        if announce_count is not None:
            self.narrator.announce_count(announce_count)

        return snapshot

    def get_snapshot(self, explored_cells):
        with self.lock:
            return {
                "victim_count": self.victim_count,
                "fire_detected": self.fire_detected,
                "explored_cells": explored_cells,
            }

    def stop(self):
        self.narrator.stop()
