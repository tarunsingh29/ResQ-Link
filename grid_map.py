import threading

import cv2
import numpy as np

from config import Config


class GridMap:
    """Grid-based coverage tracker driven by executed movement steps."""

    def __init__(self, rows=None, cols=None, scale=None):
        self.rows = rows or Config.GRID_ROWS
        self.cols = cols or Config.GRID_COLS
        self.scale = scale or Config.HEATMAP_SCALE
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.visits = np.zeros((self.rows, self.cols), dtype=np.uint16)
            self.hazards = np.zeros((self.rows, self.cols), dtype=bool)
            self.row = self.rows // 2
            self.col = self.cols // 2
            self._mark_current_locked()

    def _mark_current_locked(self):
        self.visits[self.row, self.col] += 1

    def mark_current(self):
        with self.lock:
            self._mark_current_locked()

    def apply_step(self, action):
        with self.lock:
            if action == "forward":
                self.row = max(0, self.row - 1)
            elif action == "back":
                self.row = min(self.rows - 1, self.row + 1)
            elif action == "right":
                self.col = min(self.cols - 1, self.col + 1)
            elif action == "left":
                self.col = max(0, self.col - 1)

            self._mark_current_locked()

    def mark_hazard(self):
        with self.lock:
            self.hazards[self.row, self.col] = True

    def explored_cells_count(self):
        with self.lock:
            return int(np.count_nonzero(self.visits))

    def get_heatmap(self):
        with self.lock:
            visits = self.visits.copy()
            hazards = self.hazards.copy()
            current_row = self.row
            current_col = self.col

        if visits.max() == 0:
            normalized = np.zeros_like(visits, dtype=np.uint8)
        else:
            normalized = (visits / visits.max() * 255).astype(np.uint8)

        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        heatmap = cv2.resize(
            heatmap,
            (self.cols * self.scale, self.rows * self.scale),
            interpolation=cv2.INTER_NEAREST,
        )

        for row in range(1, self.rows):
            y = row * self.scale
            cv2.line(heatmap, (0, y), (self.cols * self.scale, y), (35, 35, 35), 1)
        for col in range(1, self.cols):
            x = col * self.scale
            cv2.line(heatmap, (x, 0), (x, self.rows * self.scale), (35, 35, 35), 1)

        hazard_positions = np.argwhere(hazards)
        for row, col in hazard_positions:
            x1 = col * self.scale
            y1 = row * self.scale
            x2 = x1 + self.scale
            y2 = y1 + self.scale
            cv2.rectangle(heatmap, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cx1 = current_col * self.scale
        cy1 = current_row * self.scale
        cx2 = cx1 + self.scale
        cy2 = cy1 + self.scale
        cv2.rectangle(heatmap, (cx1, cy1), (cx2, cy2), (255, 255, 255), 2)

        return heatmap
