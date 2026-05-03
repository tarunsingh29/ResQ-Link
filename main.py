import json
import math
import sys
import time
from pathlib import Path

import cv2
from djitellopy import Tello

from config import Config
from control.demo_script import DemoScript
from controller import Controller
from detector import Detector
from frame_grabber import FrameGrabber
from grid_map import GridMap
from map_memory import MapMemory
from position_tracker import PositionTracker
from radar import RadarSystem
from renderer import Renderer
from utils.metrics import MetricsTracker

try:
    import msvcrt
except ImportError:
    msvcrt = None


next_victim_id = 1
counted_victim_ids = set()


def _read_console_key():
    """Read a key from the Windows console when GUI input is unavailable."""
    if msvcrt and msvcrt.kbhit():
        key = msvcrt.getch()
        if key in (b"\x00", b"\xe0"):
            if msvcrt.kbhit():
                msvcrt.getch()
            return -1, False
        if key == b"\x1b":
            return 27, False
        return ord(key.lower()), False

    return -1, False


def _display_frame_and_read_key(window_name, frame, gui_enabled):
    """Display the frame and read keyboard input from the active UI."""
    if not gui_enabled:
        key, _ = _read_console_key()
        return key, False

    try:
        cv2.imshow(window_name, frame)
        key = cv2.waitKeyEx(1)
        return key & 0xFF, True
    except cv2.error:
        print("[WARNING] OpenCV HighGUI display is unavailable. Dashboard window disabled.")
        print("[INFO] Console controls remain active in the terminal.")
        key, _ = _read_console_key()
        return key, False


def _destroy_windows(gui_enabled):
    """Close OpenCV windows only when HighGUI support exists."""
    if not gui_enabled:
        return

    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def _write_dashboard_state(snapshot, controller, heatmap):
    state_payload = {
        "mode": controller.mode,
        "state": controller.state,
        "victim_count": snapshot["victim_count"],
        "fire_detected": snapshot["fire_detected"],
        "explored_cells": snapshot["explored_cells"],
        "updated_at": time.time(),
    }
    Path(Config.DASHBOARD_STATE_PATH).write_text(json.dumps(state_payload, indent=2), encoding="utf-8")
    cv2.imwrite(Config.DASHBOARD_HEATMAP_PATH, heatmap)


def _area_ratio(area_a, area_b):
    return min(area_a, area_b) / max(area_a, area_b, 1.0)


def _estimate_world_position(cx, area, frame_width, position_tracker):
    if position_tracker is None or frame_width <= 0:
        return None, None

    angle = (cx - (frame_width / 2.0)) / frame_width * 90.0
    distance = max(20.0, min(300.0, 50000.0 / (area + 1.0)))

    try:
        world_x, world_y = position_tracker.project_detection(angle, distance)
    except Exception:
        return None, None

    return float(world_x), float(world_y)


def _smooth_track(track, x, y, area, current_time, world_x=None, world_y=None):
    track["x"] = int(round((0.7 * track["x"]) + (0.3 * x)))
    track["y"] = int(round((0.7 * track["y"]) + (0.3 * y)))
    track["area"] = (0.7 * track["area"]) + (0.3 * area)
    if world_x is not None and world_y is not None:
        previous_world_x = track.get("world_x", world_x)
        previous_world_y = track.get("world_y", world_y)
        track["world_x"] = (0.7 * previous_world_x) + (0.3 * world_x)
        track["world_y"] = (0.7 * previous_world_y) + (0.3 * world_y)
    track["last_seen"] = current_time


def _find_best_victim_match(
    victims,
    cx,
    cy,
    area,
    current_time,
    match_distance,
    min_area_ratio,
    world_x=None,
    world_y=None,
    world_match_distance=None,
    max_age_seconds=None,
    exclude_ids=None,
):
    excluded_ids = exclude_ids or set()
    best_victim = None
    best_score = float("inf")

    for victim in victims:
        if victim["id"] in excluded_ids:
            continue

        if max_age_seconds is not None and (current_time - victim["last_seen"]) >= max_age_seconds:
            continue

        if _area_ratio(area, victim["area"]) <= min_area_ratio:
            continue

        image_distance = math.hypot(cx - victim["x"], cy - victim["y"])
        image_match = image_distance < match_distance

        victim_world_x = victim.get("world_x")
        victim_world_y = victim.get("world_y")
        world_distance = None
        world_match = False
        if (
            world_x is not None
            and world_y is not None
            and victim_world_x is not None
            and victim_world_y is not None
            and world_match_distance is not None
        ):
            world_distance = math.hypot(world_x - victim_world_x, world_y - victim_world_y)
            world_match = world_distance < world_match_distance

        if not image_match and not world_match:
            continue

        if world_match and world_distance is not None:
            score = world_distance / max(world_match_distance, 1.0)
            if image_match:
                score += image_distance / max(match_distance, 1.0)
        else:
            score = 2.0 + (image_distance / max(match_distance, 1.0))

        if score < best_score:
            best_victim = victim
            best_score = score

    return best_victim


def _find_best_candidate_key(
    temporary_candidates,
    cx,
    cy,
    area,
    match_distance,
    min_area_ratio,
    frame_index,
    used_keys,
    world_x=None,
    world_y=None,
    world_match_distance=None,
):
    best_key = None
    best_score = float("inf")

    for key, candidate in temporary_candidates.items():
        if key in used_keys:
            continue

        if candidate.get("last_seen_frame") != (frame_index - 1):
            continue

        if _area_ratio(area, candidate["area"]) <= min_area_ratio:
            continue

        image_distance = math.hypot(cx - candidate["x"], cy - candidate["y"])
        image_match = image_distance < match_distance

        candidate_world_x = candidate.get("world_x")
        candidate_world_y = candidate.get("world_y")
        world_distance = None
        world_match = False
        if (
            world_x is not None
            and world_y is not None
            and candidate_world_x is not None
            and candidate_world_y is not None
            and world_match_distance is not None
        ):
            world_distance = math.hypot(world_x - candidate_world_x, world_y - candidate_world_y)
            world_match = world_distance < world_match_distance

        if not image_match and not world_match:
            continue

        if world_match and world_distance is not None:
            score = world_distance / max(world_match_distance, 1.0)
            if image_match:
                score += image_distance / max(match_distance, 1.0)
        else:
            score = 2.0 + (image_distance / max(match_distance, 1.0))

        if score < best_score:
            best_key = key
            best_score = score

    return best_key


def _update_tracked_victims(
    tracked_victims,
    known_victims,
    temporary_candidates,
    detections,
    current_time,
    frame_index,
    match_distance,
    track_ttl,
    frame_width,
    position_tracker,
):
    global next_victim_id

    recent_match_window = getattr(Config, "VICTIM_RECENT_MATCH_WINDOW", 3.0)
    world_match_distance = getattr(Config, "VICTIM_WORLD_MATCH_DISTANCE", 90.0)
    min_area_ratio = getattr(Config, "VICTIM_MIN_AREA_RATIO", 0.6)
    promotion_frames = getattr(Config, "VICTIM_PROMOTION_FRAMES", 2)
    min_confidence = getattr(Config, "VICTIM_MIN_TRACK_CONFIDENCE", 0.35)
    min_area = getattr(Config, "VICTIM_MIN_TRACK_AREA", 400.0)

    tracked_victims[:] = [
        victim for victim in tracked_victims
        if (current_time - victim["last_seen"]) <= track_ttl
    ]

    stale_candidate_keys = [
        key for key, candidate in temporary_candidates.items()
        if (current_time - candidate["last_seen"]) > recent_match_window
        or candidate.get("last_seen_frame", -1) < (frame_index - 1)
    ]
    for key in stale_candidate_keys:
        temporary_candidates.pop(key, None)

    new_ids = []
    active_victim_ids = {victim["id"] for victim in tracked_victims}
    used_victim_ids = set()
    used_candidate_keys = set()

    for detection in sorted(
        (d for d in detections if d.get("class") == "victim"),
        key=lambda item: float(item.get("conf") or 0.0),
        reverse=True,
    ):
        x1, y1, x2, y2 = detection["bbox"]
        area = float(max(0, x2 - x1) * max(0, y2 - y1))
        confidence = float(detection.get("conf") or 0.0)

        if area < min_area or confidence < min_confidence:
            continue

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        world_x, world_y = _estimate_world_position(cx, area, frame_width, position_tracker)

        matched_victim = _find_best_victim_match(
            victims=tracked_victims,
            cx=cx,
            cy=cy,
            area=area,
            current_time=current_time,
            match_distance=match_distance,
            min_area_ratio=min_area_ratio,
            world_x=world_x,
            world_y=world_y,
            world_match_distance=world_match_distance,
            max_age_seconds=recent_match_window,
            exclude_ids=used_victim_ids,
        )

        if matched_victim is not None:
            _smooth_track(matched_victim, cx, cy, area, current_time, world_x=world_x, world_y=world_y)
            known_victims[matched_victim["id"]] = matched_victim
            used_victim_ids.add(matched_victim["id"])
            print(f"[MATCH] Victim ID={matched_victim['id']}")
            continue

        inactive_known_victims = [
            victim for victim_id, victim in known_victims.items()
            if victim_id not in active_victim_ids
        ]
        known_match = _find_best_victim_match(
            victims=inactive_known_victims,
            cx=cx,
            cy=cy,
            area=area,
            current_time=current_time,
            match_distance=match_distance,
            min_area_ratio=min_area_ratio,
            world_x=world_x,
            world_y=world_y,
            world_match_distance=world_match_distance,
            exclude_ids=used_victim_ids,
        )

        if known_match is not None:
            _smooth_track(known_match, cx, cy, area, current_time, world_x=world_x, world_y=world_y)
            tracked_victims.append(known_match)
            active_victim_ids.add(known_match["id"])
            used_victim_ids.add(known_match["id"])
            print(f"[MATCH] Victim ID={known_match['id']}")
            print(f"[SKIP] Already counted ID={known_match['id']}")
            continue

        candidate_key = _find_best_candidate_key(
            temporary_candidates=temporary_candidates,
            cx=cx,
            cy=cy,
            area=area,
            match_distance=match_distance,
            min_area_ratio=min_area_ratio,
            frame_index=frame_index,
            used_keys=used_candidate_keys,
            world_x=world_x,
            world_y=world_y,
            world_match_distance=world_match_distance,
        )

        if candidate_key is None:
            new_candidate_key = (cx, cy)
            temporary_candidates[new_candidate_key] = {
                "x": float(cx),
                "y": float(cy),
                "area": area,
                "world_x": world_x,
                "world_y": world_y,
                "frames_seen": 1,
                "last_seen": current_time,
                "last_seen_frame": frame_index,
            }
            used_candidate_keys.add(new_candidate_key)
            continue

        candidate = temporary_candidates.pop(candidate_key)
        candidate["x"] = (0.7 * candidate["x"]) + (0.3 * cx)
        candidate["y"] = (0.7 * candidate["y"]) + (0.3 * cy)
        candidate["area"] = (0.7 * candidate["area"]) + (0.3 * area)
        if world_x is not None and world_y is not None:
            previous_world_x = candidate.get("world_x", world_x)
            previous_world_y = candidate.get("world_y", world_y)
            candidate["world_x"] = (0.7 * previous_world_x) + (0.3 * world_x)
            candidate["world_y"] = (0.7 * previous_world_y) + (0.3 * world_y)
        candidate["frames_seen"] += 1
        candidate["last_seen"] = current_time
        candidate["last_seen_frame"] = frame_index

        if candidate["frames_seen"] < promotion_frames:
            new_candidate_key = (int(round(candidate["x"])), int(round(candidate["y"])))
            temporary_candidates[new_candidate_key] = candidate
            used_candidate_keys.add(new_candidate_key)
            continue

        victim_id = next_victim_id
        victim = {
            "id": victim_id,
            "x": int(round(candidate["x"])),
            "y": int(round(candidate["y"])),
            "area": candidate["area"],
            "world_x": candidate.get("world_x"),
            "world_y": candidate.get("world_y"),
            "last_seen": current_time,
        }
        next_victim_id += 1
        tracked_victims.append(victim)
        known_victims[victim_id] = victim
        active_victim_ids.add(victim_id)
        used_victim_ids.add(victim_id)
        new_ids.append(victim_id)

    return new_ids


def main():
    global next_victim_id, counted_victim_ids

    print("[INFO] Initializing GUARDIAN System...")

    tello = Tello()
    try:
        tello.connect()
        try:
            tello.set_video_fps(Tello.FPS_30)
        except Exception:
            pass
        try:
            tello.set_speed(Config.TELLO_SPEED)
        except Exception:
            pass

        try:
            tello.streamoff()
            time.sleep(0.5)
        except Exception:
            pass

        tello.streamon()
        time.sleep(1.0)
        battery = tello.get_battery()
        print(f"[INFO] Tello Connected. Battery: {battery}%")

        if battery < 10:
            print("[WARNING] Battery is very low!")
    except Exception as exc:
        print(f"[ERROR] Could not connect to Tello: {exc}")
        print("Please check Wi-Fi connection to the drone.")
        sys.exit(1)

    grabber = FrameGrabber(tello).start()
    detector = Detector().start()
    grid_map = GridMap()
    metrics = MetricsTracker()
    position_tracker = PositionTracker()
    radar = RadarSystem()
    map_memory = MapMemory()
    controller = Controller(tello, grid_map, demo_script=DemoScript(), position_tracker=position_tracker)

    print("[INFO] Waiting for video stream...")
    while grabber.get_frame() is None:
        time.sleep(0.1)

    print("[INFO] Stream started. System READY.")
    print("--------------------------------------------------")
    print(" CONTROLS:")
    print(" 't' - Takeoff        'l' - Land")
    print(" 'y' - Toggle ROTATION_SCAN mode")
    print(" 'u' - Toggle DEMO_SCRIPT mode")
    print(" 'w','s','a','d' - Discrete manual move (MANUAL mode)")
    print(" 'r','f' - Manual altitude up/down")
    print(" 'x' - Hover / hold current position")
    print(" 'q' or ESC - Quit")
    print("--------------------------------------------------")

    last_loop_time = time.time()
    last_battery_time = 0.0
    last_detection_log_time = 0.0
    last_detection_signature = None
    last_dashboard_update_time = 0.0
    battery_display = 0
    gui_enabled = True
    window_name = "GUARDIAN Dashboard"
    detection_interval = max(1, getattr(Config, "DETECTION_INTERVAL", 4))
    dashboard_update_interval = max(Config.DASHBOARD_UPDATE_INTERVAL, 1.5)
    frame_counter = 0
    tracking_frame_index = 0
    tracked_victims = []
    known_victims = {}
    temporary_candidates = {}
    victim_track_ttl = getattr(Config, "VICTIM_TRACK_TTL", 8.0)
    victim_total = 0
    last_announced_victim_count = 0
    next_victim_id = 1
    counted_victim_ids.clear()

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT)
        cv2.moveWindow(window_name, 50, 50)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        print("[INFO] Video stream opens in a separate OpenCV window.")
    except cv2.error:
        gui_enabled = False
        print("[WARNING] OpenCV GUI window could not be created.")
        print("[INFO] If you installed a headless OpenCV build, reinstall `opencv-python`.")

    try:
        while True:
            frame = grabber.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            if controller.mode == "ROTATION_SCAN":
                controller.update()

            should_run_detection = (frame_counter % detection_interval) == 0
            frame_counter += 1
            if should_run_detection:
                detector.update_frame(frame)
            detections, inf_fps = detector.get_detections()
            if should_run_detection:
                fresh_radar_points = radar.update(detections, frame.shape[1])
                for point in fresh_radar_points:
                    obj_x, obj_y = position_tracker.project_detection(point["angle"], point["distance"])
                    map_memory.add_object(
                        obj_x,
                        obj_y,
                        point["label"],
                        confidence=point.get("confidence", 1.0),
                    )
            radar.decay()
            radar_points = radar.get_points()

            detection_signature = tuple(
                (d["class"], round(d["conf"], 2), tuple(d["bbox"]))
                for d in detections
            )
            if detections and (
                detection_signature != last_detection_signature
                or (time.time() - last_detection_log_time) >= Config.LOG_DETECTION_INTERVAL
            ):
                print("\n--- DETECTIONS ---")
                for detection in detections:
                    print(
                        f"{detection['class'].upper()} | conf: {round(detection['conf'], 2)} | "
                        f"bbox: {detection['bbox']}"
                    )
                last_detection_log_time = time.time()
                last_detection_signature = detection_signature
            elif not detections:
                last_detection_signature = None

            curr_time = time.time()
            main_fps = 1.0 / (curr_time - last_loop_time + 1e-5)
            last_loop_time = curr_time

            if should_run_detection:
                tracking_frame_index += 1
                new_ids = _update_tracked_victims(
                    tracked_victims=tracked_victims,
                    known_victims=known_victims,
                    temporary_candidates=temporary_candidates,
                    detections=detections,
                    current_time=curr_time,
                    frame_index=tracking_frame_index,
                    match_distance=Config.VICTIM_MATCH_DISTANCE,
                    track_ttl=victim_track_ttl,
                    frame_width=frame.shape[1],
                    position_tracker=position_tracker,
                )
                for victim_id in new_ids:
                    if victim_id not in counted_victim_ids:
                        counted_victim_ids.add(victim_id)
                        victim_total += 1
                        print(f"[NEW] Victim ID={victim_id} (COUNTED)")
                    else:
                        print(f"[SKIP] Already counted ID={victim_id}")
            metrics.update([d for d in detections if d.get("class") != "victim"])
            fire_detected = any(d.get("class") == "fire" for d in detections)

            if fire_detected:
                grid_map.mark_hazard()

                fire_close = False
                for d in detections:
                    if d.get("class") == "fire":
                        x1, y1, x2, y2 = d["bbox"]
                        area = (x2 - x1) * (y2 - y1)

                        if area >= Config.FIRE_MIN_AREA:
                            fire_close = True
                            break

                if fire_close:
                    controller.trigger_fire_escape()
            current_metrics = metrics.get_snapshot(grid_map.explored_cells_count())
            current_metrics["victim_count"] = victim_total

            if victim_total > last_announced_victim_count:
                metrics.narrator.announce_count(victim_total)
                last_announced_victim_count = victim_total

            if controller.is_flying and controller.mode != "ROTATION_SCAN":
                controller.update()

            if time.time() - last_battery_time > 5:
                try:
                    battery_display = tello.get_battery()
                except Exception:
                    pass
                last_battery_time = time.time()

            display_frame = Renderer.draw_hud(
                frame=frame,
                state=controller.state,
                mode=controller.mode,
                battery=battery_display,
                detections=detections,
                metrics=current_metrics,
                main_fps=main_fps,
                inference_fps=inf_fps,
            )
            Renderer.draw_spatial_map(display_frame, position_tracker, map_memory)
            Renderer.draw_radar(display_frame, radar_points)

            if (time.time() - last_dashboard_update_time) >= dashboard_update_interval:
                _write_dashboard_state(current_metrics, controller, grid_map.get_heatmap())
                last_dashboard_update_time = time.time()

            key, gui_enabled = _display_frame_and_read_key(window_name, display_frame, gui_enabled)

            if key == 27 or key == ord("q"):
                break
            if key == ord("t") and not controller.is_flying:
                print("[INFO] Taking off...")
                tello.takeoff()
                grid_map.reset()
                metrics.reset()
                position_tracker.reset()
                radar.reset()
                map_memory.reset()
                tracked_victims.clear()
                known_victims.clear()
                temporary_candidates.clear()
                victim_total = 0
                last_announced_victim_count = 0
                tracking_frame_index = 0
                next_victim_id = 1
                counted_victim_ids.clear()
                controller.on_takeoff()
            elif key == ord("l") and controller.is_flying:
                if controller.is_busy():
                    print("[INFO] Waiting for the current movement to finish before landing.")
                    continue
                print("[INFO] Landing...")
                controller._hard_stop()
                tello.land()
                controller.on_land()
            elif key == ord("y"):
                controller.toggle_rotation_scan()
                print(f"[INFO] Mode Switched to: {controller.mode}")
            elif key == ord("u"):
                controller.toggle_demo_script()
                print(f"[INFO] Mode Switched to: {controller.mode}")
            elif key == ord("x"):
                controller.hover()
            elif key == ord("w"):
                controller.request_manual_step("forward")
            elif key == ord("s"):
                controller.request_manual_step("back")
            elif key == ord("a"):
                controller.request_manual_step("left")
            elif key == ord("d"):
                controller.request_manual_step("right")
            elif key == ord("r"):
                controller.request_manual_step("up")
            elif key == ord("f"):
                controller.request_manual_step("down")

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt by user.")
    except Exception as exc:
        print(f"\n[ERROR] Main loop exception: {exc}")
    finally:
        print("[INFO] Cleaning up system...")
        if controller.is_flying:
            try:
                controller._hard_stop()
                tello.land()
            except Exception:
                pass
        metrics.stop()
        grabber.stop()
        detector.stop()
        try:
            tello.streamoff()
        except Exception:
            pass
        _destroy_windows(gui_enabled)
        print("[INFO] Shutdown complete.")


if __name__ == "__main__":
    main()
