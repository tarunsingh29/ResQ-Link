from config import Config


class DemoScript:
    """Fixed, repeatable exploration sequence for demo runs."""

    def __init__(self, sequence=None):
        self.sequence = list(sequence or Config.DEMO_SCRIPT_SEQUENCE)
        self.index = 0

    def reset(self):
        self.index = 0

    def _build_step(self, action):
        if action == "forward":
            return {
                "action": "forward",
                "value": Config.DEMO_STEP_CM,
                "state": "DEMO_FORWARD",
                "wait_after": Config.COMMAND_DELAY,
                "grid_action": "forward",
            }

        if action == "rotate":
            return {
                "action": "rotate_clockwise",
                "value": 90,
                "state": "DEMO_ROTATE",
                "wait_after": 1.5,
                "grid_action": None,
            }

        raise ValueError(f"Unsupported demo action: {action}")

    def peek_step(self):
        if not self.sequence:
            return None

        action = self.sequence[self.index]
        return self._build_step(action)

    def advance(self):
        if not self.sequence:
            return

        self.index = (self.index + 1) % len(self.sequence)

    def next_step(self):
        step = self.peek_step()
        if step is None:
            return None

        self.advance()
        return step
