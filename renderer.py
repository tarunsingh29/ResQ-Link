import math
import time

import cv2
from config import Config

class Renderer:
    """Draws the clean dashboard UI over the frame."""
    
    @staticmethod
    def draw_hud(frame, state, mode, battery, detections, metrics, main_fps, inference_fps):
        if frame.shape[1] != Config.CAMERA_WIDTH or frame.shape[0] != Config.CAMERA_HEIGHT:
            frame = cv2.resize(frame, (Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT))

        #  Draw Detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['conf']
            is_fire = det.get('class') == "fire"
            box_color = Config.COLOR_WARNING if is_fire else Config.COLOR_BBOX
            
            # Draw Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw Label with dark background for readability
            label = f"{det['class'].upper()} {int(conf * 100)}%"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - 25)), (x1 + w, max(0, y1 - 25) + h + 10), Config.COLOR_BG, -1)
            cv2.putText(frame, label, (x1, max(20, y1 - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            # Draw crosshair over target center
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            marker_color = Config.COLOR_WARNING if is_fire else Config.COLOR_INFO
            cv2.drawMarker(frame, (cx, cy), marker_color, cv2.MARKER_CROSS, 20, 2)

        #  Draw Top Left Panel (Mode & State)
        mode_color = Config.COLOR_WARNING if mode != 'MANUAL' else Config.COLOR_TEXT
        cv2.putText(frame, f"MODE: {mode}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)
                    
        cv2.putText(frame, f"STATE: {state}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, Config.COLOR_INFO, 2)

        victim_count = metrics.get("victim_count", 0)
        fire_status = "YES" if metrics.get("fire_detected") else "NO"
        explored_cells = metrics.get("explored_cells", 0)
        cv2.putText(frame, f"VICTIMS: {victim_count}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, Config.COLOR_BBOX, 2)
        cv2.putText(frame, f"FIRE: {fire_status}", (20, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, Config.COLOR_WARNING, 2)
        cv2.putText(frame, f"COVERAGE: {explored_cells}", (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, Config.COLOR_TEXT, 2)
                    
        #  Draw Bottom Left Stats (FPS)
        h_frame = Config.CAMERA_HEIGHT
        cv2.putText(frame, f"SYS FPS: {main_fps:.1f}", (20, h_frame - 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.COLOR_TEXT, 2)
        cv2.putText(frame, f"YOLO FPS: {inference_fps:.1f}", (20, h_frame - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.COLOR_TEXT, 2)
                    
        #  Draw Top Right Battery
        bat_color = Config.COLOR_WARNING if battery < 20 else Config.COLOR_BBOX
        # Align battery to right
        bat_str = f"BAT: {battery}%"
        (bw, _), _ = cv2.getTextSize(bat_str, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.putText(frame, bat_str, (Config.CAMERA_WIDTH - bw - 20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, bat_color, 2)
                    
        #  Draw Static Camera Center Crosshair
        ccx, ccy = Config.CAMERA_WIDTH // 2, Config.CAMERA_HEIGHT // 2
        cv2.drawMarker(frame, (ccx, ccy), Config.COLOR_TEXT, cv2.MARKER_CROSS, 40, 1)
        
        return frame

    @staticmethod
    def draw_radar(frame, radar_points):
        radius = 105
        padding = 30
        center = (Config.CAMERA_WIDTH - radius - padding, Config.CAMERA_HEIGHT - radius - padding)
        min_distance = 20.0
        max_distance = 300.0
        current_time = time.time()

        cv2.circle(frame, center, radius + 8, (10, 10, 10), -1)
        cv2.circle(frame, center, radius, Config.COLOR_TEXT, 2)

        ring_distances = (75, 150, 225, 300)
        for ring_distance in ring_distances:
            ring_ratio = (ring_distance - min_distance) / (max_distance - min_distance)
            ring_radius = max(12, int(ring_ratio * radius))
            cv2.circle(frame, center, ring_radius, (55, 55, 55), 1)
            cv2.putText(
                frame,
                str(ring_distance),
                (center[0] + 8, center[1] - ring_radius + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (120, 120, 120),
                1,
            )

        for guide_angle in (-45, 0, 45):
            theta = math.radians(guide_angle - 90)
            guide_x = int(center[0] + radius * math.cos(theta))
            guide_y = int(center[1] + radius * math.sin(theta))
            cv2.line(frame, center, (guide_x, guide_y), (45, 45, 45), 1)

        sweep_angle = (current_time * 140.0) % 360.0
        for offset, color_scale in ((0.0, 1.0), (8.0, 0.55), (16.0, 0.25)):
            theta = math.radians((sweep_angle - offset) - 90.0)
            sweep_x = int(center[0] + radius * math.cos(theta))
            sweep_y = int(center[1] + radius * math.sin(theta))
            sweep_color = (int(40 * color_scale), int(110 * color_scale), int(255 * color_scale))
            cv2.line(frame, center, (sweep_x, sweep_y), sweep_color, 2 if offset == 0.0 else 1)

        cv2.putText(
            frame,
            "RADAR",
            (center[0] - radius + 4, center[1] - radius - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            Config.COLOR_TEXT,
            2,
        )

        for point in sorted(radar_points, key=lambda item: item["timestamp"]):
            age = current_time - point["timestamp"]
            fade = max(0.15, 1.0 - (age / 3.0))
            distance = max(min_distance, min(max_distance, point["distance"]))
            distance_ratio = (distance - min_distance) / (max_distance - min_distance)
            radial_distance = max(10, int(distance_ratio * radius))
            theta = math.radians(point["angle"] - 90.0)
            point_x = int(center[0] + radial_distance * math.cos(theta))
            point_y = int(center[1] + radial_distance * math.sin(theta))

            if point["label"] == "fire":
                base_color = (0, 0, 255)
            elif point["label"] == "victim":
                base_color = (0, 255, 255)
            else:
                base_color = Config.COLOR_INFO

            color = tuple(int(channel * fade) for channel in base_color)
            marker_radius = 4 + int(point.get("confidence", 0.0) * 3)
            cv2.circle(frame, (point_x, point_y), marker_radius, color, -1)
            cv2.circle(frame, (point_x, point_y), marker_radius + 2, color, 1)

        cv2.circle(frame, center, 5, Config.COLOR_INFO, -1)
        cv2.circle(frame, center, 8, Config.COLOR_TEXT, 1)
        return frame

    @staticmethod
    def draw_spatial_map(frame, position_tracker, map_memory):
        if position_tracker is None or map_memory is None:
            return frame

        panel_size = 180
        padding = 28
        top = 70
        left = Config.CAMERA_WIDTH - panel_size - padding
        right = left + panel_size
        bottom = top + panel_size
        inner_padding = 14

        cv2.rectangle(frame, (left, top), (right, bottom), (12, 12, 12), -1)
        cv2.rectangle(frame, (left, top), (right, bottom), Config.COLOR_TEXT, 2)
        cv2.putText(
            frame,
            "MAP",
            (left + 8, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            Config.COLOR_TEXT,
            2,
        )

        objects = map_memory.get_objects()
        path_points = list(position_tracker.path_history)
        current_pose = (position_tracker.x, position_tracker.y)
        all_points = path_points + [current_pose] + [(obj["x"], obj["y"]) for obj in objects]

        if not all_points:
            return frame

        xs = [point[0] for point in all_points]
        ys = [point[1] for point in all_points]
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)

        span = max(max_x - min_x, max_y - min_y, 200.0)
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        half_span = (span / 2.0) + 40.0

        draw_left = left + inner_padding
        draw_top = top + inner_padding
        draw_right = right - inner_padding
        draw_bottom = bottom - inner_padding
        draw_size = min(draw_right - draw_left, draw_bottom - draw_top)
        scale = draw_size / (half_span * 2.0)

        def to_panel(point_x, point_y):
            mapped_x = int((point_x - (center_x - half_span)) * scale) + draw_left
            mapped_y = draw_bottom - int((point_y - (center_y - half_span)) * scale)
            return mapped_x, mapped_y

        for grid_index in range(1, 4):
            fraction = grid_index / 4.0
            grid_x = int(draw_left + (draw_size * fraction))
            grid_y = int(draw_top + (draw_size * fraction))
            cv2.line(frame, (grid_x, draw_top), (grid_x, draw_bottom), (40, 40, 40), 1)
            cv2.line(frame, (draw_left, grid_y), (draw_right, grid_y), (40, 40, 40), 1)

        origin_x, origin_y = to_panel(0.0, 0.0)
        cv2.drawMarker(frame, (origin_x, origin_y), (90, 90, 90), cv2.MARKER_TILTED_CROSS, 12, 1)

        if len(path_points) >= 2:
            for start, end in zip(path_points[:-1], path_points[1:]):
                cv2.line(frame, to_panel(*start), to_panel(*end), (80, 200, 255), 2)

        for obj in objects:
            if obj["type"] == "fire":
                color = (0, 0, 255)
            elif obj["type"] == "victim":
                color = (0, 255, 255)
            else:
                color = Config.COLOR_INFO

            point_x, point_y = to_panel(obj["x"], obj["y"])
            cv2.circle(frame, (point_x, point_y), 5, color, -1)
            cv2.circle(frame, (point_x, point_y), 8, color, 1)

        drone_x, drone_y = to_panel(*current_pose)
        heading = math.radians(position_tracker.yaw)
        arrow_length = 18
        arrow_tip = (
            int(drone_x + arrow_length * math.cos(heading)),
            int(drone_y - arrow_length * math.sin(heading)),
        )
        cv2.circle(frame, (drone_x, drone_y), 6, Config.COLOR_INFO, -1)
        cv2.arrowedLine(frame, (drone_x, drone_y), arrow_tip, Config.COLOR_INFO, 2, tipLength=0.35)

        cv2.putText(
            frame,
            f"OBJ {len(objects)}",
            (left + 8, bottom + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            Config.COLOR_TEXT,
            1,
        )
        return frame
