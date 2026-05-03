from config import Config


def clamp_speed(speed):
    """Ensure speed is within the allowed bounds."""
    return max(-Config.MAX_SPEED, min(Config.MAX_SPEED, int(speed)))


def get_center(bbox):
    """Return the (x, y) center of a bounding box."""
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_area(bbox):
    """Return the area of a bounding box."""
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)
