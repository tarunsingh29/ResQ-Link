import math


class MapMemory:
    """Stores and merges detected objects on the map."""

    def __init__(self, merge_threshold=50.0):
        self.objects = []
        self.merge_threshold = merge_threshold

    def reset(self):
        self.objects = []

    def add_object(self, x, y, obj_type, confidence=1.0):
        """Add an object to the map, merging if close to existing objects."""
        new_obj = {"x": x, "y": y, "type": obj_type, "confidence": confidence}

        closest_object = None
        closest_distance = None
        for obj in self.objects:
            if obj["type"] == obj_type:
                distance = math.hypot(x - obj["x"], y - obj["y"])
                if distance < self.merge_threshold and (
                    closest_distance is None or distance < closest_distance
                ):
                    closest_object = obj
                    closest_distance = distance

        if closest_object is None:
            self.objects.append(new_obj)
            return

        existing_weight = max(closest_object["confidence"], 0.1)
        new_weight = max(confidence, 0.1)
        total_weight = existing_weight + new_weight
        closest_object["x"] = ((closest_object["x"] * existing_weight) + (x * new_weight)) / total_weight
        closest_object["y"] = ((closest_object["y"] * existing_weight) + (y * new_weight)) / total_weight
        closest_object["confidence"] = max(closest_object["confidence"], confidence)

    def get_objects(self):
        return [obj.copy() for obj in self.objects]
