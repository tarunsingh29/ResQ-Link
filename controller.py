import threading
import time

from control.demo_script import DemoScript
from config import Config


class Controller:
    """Deterministic controller with manual, scan, and scripted demo modes."""

    def __init__(self, tello, grid_map, demo_script=None, position_tracker=None):
        self.tello = tello
        self.grid_map = grid_map
        self.demo_script = demo_script or DemoScript()
        self.position_tracker = position_tracker
        self.mode = "MANUAL"
        self.state = "IDLE"
        self.last_command_completed = 0.0
        self.is_flying = False
        self.command_in_progress = False
        self.command_worker_active = False
        self.command_lock = threading.Lock()
        self.last_error = None
        self.rc_active = False
        self.current_rc = (0, 0, 0, 0)
        self.last_rc_time = 0.0
        self.RC_INTERVAL = 0.08
        self.needs_stabilization = False
        self.last_rotation_scan_update = None
        self.fire_escape_active = False
        self.prev_mode = None
        self.last_fire_escape_time = 0.0

    def _log_control(self, message):
        print(f"[CONTROL] {message}")

    def _set_rc(self, lr, fb, ud, yaw):
        cmd = (lr, fb, ud, yaw)
        allow_repeat_scan_rc = self.mode == "ROTATION_SCAN" and any(cmd)

        if cmd == self.current_rc and not allow_repeat_scan_rc:
            return

        try:
            was_active = self.rc_active
            self.tello.send_rc_control(lr, fb, ud, yaw)
            self.current_rc = cmd
            self.rc_active = any(cmd)

            if not was_active and self.rc_active:
                self._log_control("RC START")
            elif was_active and not self.rc_active:
                self._log_control("RC STOP")
        except Exception:
            pass

    def on_takeoff(self):
        self.is_flying = True
        self.state = "READY"
        self.last_error = None
        self.last_command_completed = time.time()
        self.command_in_progress = False
        self.command_worker_active = False
        self.rc_active = False
        self.current_rc = (0, 0, 0, 0)
        self.last_rc_time = 0.0
        self.needs_stabilization = True
        self.last_rotation_scan_update = None
        self.fire_escape_active = False
        self.prev_mode = None
        self.last_fire_escape_time = 0.0

    def on_land(self):
        self._hard_stop()
        self.is_flying = False
        self.mode = "MANUAL"
        self.state = "IDLE"
        self.last_error = None
        self.command_in_progress = False
        self.command_worker_active = False
        self.demo_script.reset()
        self.rc_active = False
        self.current_rc = (0, 0, 0, 0)
        self.last_rc_time = 0.0
        self.needs_stabilization = False
        self.last_rotation_scan_update = None
        self.fire_escape_active = False
        self.prev_mode = None
        self.last_fire_escape_time = 0.0

    def _hard_stop(self):
        self.last_rotation_scan_update = None
        self.last_rc_time = 0.0
        self._set_rc(0, 0, 0, 0)
        self.needs_stabilization = False

    def _start_rotation_rc(self):
        if not self.is_flying or self.command_worker_active:
            return self.rc_active

        self._set_rc(0, 0, 0, Config.SCAN_YAW_SPEED)
        if self.rc_active:
            self.last_rotation_scan_update = time.time()
            self.needs_stabilization = True
            return True
        return False

    def _stop_rotation_rc(self):
        self._hard_stop()

    def _stabilize_hover(self):
        if not self.is_flying or self.mode == "ROTATION_SCAN":
            return False

        with self.command_lock:
            if self.command_in_progress:
                return False

        if not self.rc_active and self.current_rc == (0, 0, 0, 0):
            return False

        self._hard_stop()
        return True

    def _set_mode(self, mode):
        previous_mode = self.mode
        self._hard_stop()

        with self.command_lock:
            self.command_in_progress = False

        self.mode = mode
        self.last_error = None
        self.last_rotation_scan_update = None

        if previous_mode == "DEMO_SCRIPT" or mode == "DEMO_SCRIPT":
            self.demo_script.reset()

        self.state = "READY" if self.is_flying else "IDLE"
        self.needs_stabilization = mode != "ROTATION_SCAN"
        self._log_control(f"MODE {previous_mode} -> {mode}")

    def toggle_rotation_scan(self):
        with self.command_lock:
            if self.fire_escape_active:
                return False

        if self.mode == "ROTATION_SCAN":
            self._set_mode("MANUAL")
        else:
            self._set_mode("ROTATION_SCAN")
        return True

    def toggle_demo_script(self):
        with self.command_lock:
            if self.fire_escape_active:
                return False

        if self.mode == "DEMO_SCRIPT":
            self._set_mode("MANUAL")
        else:
            self._set_mode("DEMO_SCRIPT")
        return True

    def hover(self):
        if not self.is_flying:
            self.state = "IDLE"
            return

        with self.command_lock:
            if self.fire_escape_active:
                return

        if self.mode != "MANUAL":
            self._set_mode("MANUAL")
        else:
            self._hard_stop()
        self.state = "HOVER"

    def is_busy(self):
        with self.command_lock:
            return self.command_in_progress or self.command_worker_active

    def _ready_for_next_step(self):
        with self.command_lock:
            return self.is_flying and not self.command_in_progress

    def _build_step(self, action, value, state, wait_after=None, grid_action=None):
        return {
            "action": action,
            "value": value,
            "state": state,
            "wait_after": Config.COMMAND_DELAY if wait_after is None else wait_after,
            "grid_action": grid_action,
        }

    def request_manual_step(self, action):
        with self.command_lock:
            if self.fire_escape_active:
                return False

        if self.mode != "MANUAL" or not self._ready_for_next_step():
            return False

        if action == "forward":
            step = self._build_step("forward", Config.MANUAL_STEP_CM, "MANUAL_FORWARD", grid_action="forward")
        elif action == "back":
            step = self._build_step("back", Config.MANUAL_STEP_CM, "MANUAL_BACK", grid_action="back")
        elif action == "left":
            step = self._build_step("left", Config.MANUAL_STEP_CM, "MANUAL_LEFT", grid_action="left")
        elif action == "right":
            step = self._build_step("right", Config.MANUAL_STEP_CM, "MANUAL_RIGHT", grid_action="right")
        elif action == "up":
            step = self._build_step("up", Config.ALTITUDE_STEP_CM, "MANUAL_UP")
        elif action == "down":
            step = self._build_step("down", Config.ALTITUDE_STEP_CM, "MANUAL_DOWN")
        else:
            return False

        return self._start_step(step)

    def trigger_fire_escape(self):
        if not self.is_flying:
            return False

        now = time.time()
        with self.command_lock:
            if self.fire_escape_active:
                return False
            if self.command_in_progress or self.command_worker_active:
                return False
            if hasattr(self, "last_fire_escape_time"):
                if now - self.last_fire_escape_time < 3.0:
                    return False

            self.fire_escape_active = True
            self.prev_mode = self.mode
            self.last_fire_escape_time = now

        self._log_control(f"FIRE ESCAPE TRIGGERED (from {self.prev_mode})")

        # Stop any scan RC motion before starting deterministic escape steps.
        self._hard_stop()

        if self.mode in ["ROTATION_SCAN", "DEMO_SCRIPT"]:
            self._log_control(f"INTERRUPTING MODE: {self.mode}")
            self.mode = "MANUAL"

        step_rotate = self._build_step(
            action="rotate_clockwise",
            value=180,
            state="FIRE_ESCAPE_ROTATE",
        )
        step_forward = self._build_step(
            action="forward",
            value=Config.DEMO_STEP_CM,
            state="FIRE_ESCAPE_FORWARD",
            grid_action="forward",
        )

        if not self._start_step(step_rotate):
            with self.command_lock:
                self.fire_escape_active = False
                self.prev_mode = None
            return False

        threading.Thread(
            target=self._chain_fire_escape,
            args=(step_forward,),
            daemon=True,
        ).start()

        return True

    def _chain_fire_escape(self, next_step):
        while True:
            time.sleep(0.05)
            with self.command_lock:
                if not self.command_in_progress:
                    break

        if self.is_flying:
            self._start_step(next_step)

            while True:
                time.sleep(0.05)
                with self.command_lock:
                    if not self.command_in_progress:
                        break

        with self.command_lock:
            self.fire_escape_active = False
            prev_mode = self.prev_mode
            self.prev_mode = None

        if not self.is_flying:
            self._log_control("FIRE ESCAPE COMPLETE")
            return

        if prev_mode in ["ROTATION_SCAN", "DEMO_SCRIPT"]:
            self._log_control(f"RESUMING MODE: {prev_mode}")
            self._set_mode(prev_mode)
        else:
            self.mode = "MANUAL"

        self._log_control("FIRE ESCAPE COMPLETE")

    def update(self):
        with self.command_lock:
            if self.fire_escape_active:
                return False

        if self.mode == "ROTATION_SCAN" and not self.is_flying:
            return False

        if not self.is_flying:
            return False

        if self.mode == "ROTATION_SCAN":
            if self.is_busy():
                return False

            self.last_error = None
            now = time.time()

            if self.position_tracker is not None and self.last_rotation_scan_update is not None:
                delta_seconds = now - self.last_rotation_scan_update
                # Approximate RC yaw speed as degrees/second for a stable fake-SLAM heading.
                self.position_tracker.rotate_by(-Config.SCAN_YAW_SPEED * delta_seconds)
            self.last_rotation_scan_update = now

            if (now - self.last_rc_time) >= self.RC_INTERVAL:
                print("[RC] sending rotation command")
                self._set_rc(0, 0, 0, Config.SCAN_YAW_SPEED)
                self.last_rc_time = now
                print("[RC] interval maintained")

            self.state = "SCAN_ROTATE"
            return True

        if self.rc_active:
            self._hard_stop()

        if self.mode == "DEMO_SCRIPT" and self._ready_for_next_step():
            step = self.demo_script.peek_step()
            if step and self._start_step(step):
                self.demo_script.advance()
                return True

        with self.command_lock:
            command_in_progress = self.command_in_progress
        if self.mode != "ROTATION_SCAN" and not command_in_progress:
            self._hard_stop()
        return False

    def _start_step(self, step):
        if step is None or not self.is_flying:
            return False

        self._hard_stop()
        with self.command_lock:
            if self.command_in_progress or self.command_worker_active:
                return False

            self.command_in_progress = True
            self.state = step["state"]
            self.last_error = None

        thread = threading.Thread(target=self._execute_step, args=(step,), daemon=True)
        thread.start()
        return True

    def _execute_step(self, step):
        mode_at_start = self.mode
        with self.command_lock:
            self.command_worker_active = True

        try:
            action = step["action"]
            value = int(step["value"])
            self._log_control(f"MOVE: {action}")

            if action == "forward":
                self.tello.move_forward(value)
            elif action == "back":
                self.tello.move_back(value)
            elif action == "left":
                self.tello.move_left(value)
            elif action == "right":
                self.tello.move_right(value)
            elif action == "up":
                self.tello.move_up(value)
            elif action == "down":
                self.tello.move_down(value)
            elif action == "rotate_clockwise":
                self.tello.rotate_clockwise(value)
            elif action == "rotate_counter_clockwise":
                self.tello.rotate_counter_clockwise(value)

            grid_action = step.get("grid_action")
            if grid_action:
                self.grid_map.apply_step(grid_action)
            else:
                self.grid_map.mark_current()

            # Update position tracker
            if self.position_tracker:
                self.position_tracker.update_from_command(action, value)
        except Exception as exc:
            self.last_error = str(exc)
            if self.mode == mode_at_start and mode_at_start in {"DEMO_SCRIPT", "ROTATION_SCAN"}:
                self.mode = "MANUAL"
                if mode_at_start == "DEMO_SCRIPT":
                    self.demo_script.reset()
            print(f"[ERROR] Command failed during {step['state']}: {exc}")
        finally:
            time.sleep(step.get("wait_after", Config.COMMAND_DELAY))
            with self.command_lock:
                self.command_worker_active = False
                self.command_in_progress = False
                self.last_command_completed = time.time()
            self.needs_stabilization = True
            if not self.is_flying:
                self.state = "IDLE"
            elif self.last_error:
                self.state = "ERROR"
            elif self.mode == "MANUAL":
                self.state = "HOVER"
            else:
                self.state = "READY"
