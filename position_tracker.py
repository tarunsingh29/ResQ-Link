import math


class PositionTracker:
    """Simulates position tracking based on drone commands."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.path_history = [(self.x, self.y)]

    def _append_path_point(self):
        self.path_history.append((self.x, self.y))

    def rotate_by(self, delta_degrees, record_path=False):
        self.yaw = (self.yaw + delta_degrees) % 360
        if record_path:
            self._append_path_point()

    def update_from_command(self, action, value):
        """Update position based on executed command."""
        if action == "forward":
            self.x += value * math.cos(math.radians(self.yaw))
            self.y += value * math.sin(math.radians(self.yaw))
        elif action == "back":
            self.x -= value * math.cos(math.radians(self.yaw))
            self.y -= value * math.sin(math.radians(self.yaw))
        elif action == "left":
            self.x += value * math.cos(math.radians(self.yaw + 90))
            self.y += value * math.sin(math.radians(self.yaw + 90))
        elif action == "right":
            self.x += value * math.cos(math.radians(self.yaw - 90))
            self.y += value * math.sin(math.radians(self.yaw - 90))
        elif action == "rotate_clockwise":
            self.rotate_by(-value)
        elif action == "rotate_counter_clockwise":
            self.rotate_by(value)

        self._append_path_point()

    def project_detection(self, angle, distance):
        """Project a detection from local radar to global map coordinates."""
        global_angle = math.radians(self.yaw + angle)
        obj_x = self.x + distance * math.cos(global_angle)
        obj_y = self.y + distance * math.sin(global_angle)
        return obj_x, obj_y
