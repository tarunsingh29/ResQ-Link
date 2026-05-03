class Config:
    # Display Settings
    CAMERA_WIDTH = 960
    CAMERA_HEIGHT = 720
    YOLO_WIDTH = 320

    # YOLO Settings
    PERSON_MODEL = "yolov8n.pt"
    FIRE_MODEL = "best.pt"

    CONFIDENCE_THRESHOLD = 0.45
    FIRE_CONFIDENCE_THRESHOLD = 0.60
    FIRE_MIN_AREA = 5000
    FIRE_COLOR_RATIO_THRESHOLD = 0.25
    FIRE_MIN_BRIGHTNESS = 120
    FIRE_VALIDATION_FRAMES = 2
    FIRE_TRACK_HISTORY = 3
    FIRE_MATCH_DISTANCE = 90
    LOG_DETECTION_INTERVAL = 2.0
    CLASS_ALIASES = {
        "victim": ("victim", "person"),
        "fire": ("fire",),
    }

    # Control Settings
    TELLO_SPEED = 20
    MAX_SPEED = 30
    COMMAND_DELAY = 1.0
    MANUAL_STEP_CM = 30
    ALTITUDE_STEP_CM = 30
    DEMO_STEP_CM = 120
    SCAN_YAW_SPEED = 20
    DEMO_SCRIPT_SEQUENCE = (
        "forward",
        "rotate",
        "forward",
        "rotate",
        "forward",
        "rotate",
        "forward",
        "rotate",
    )

    # Dashboard Settings
    GRID_ROWS = 12
    GRID_COLS = 12
    HEATMAP_SCALE = 36
    DASHBOARD_STATE_PATH = "dashboard_state.json"
    DASHBOARD_HEATMAP_PATH = "dashboard_heatmap.png"
    DASHBOARD_UPDATE_INTERVAL = 0.5

    # Metrics Settings
    VICTIM_MATCH_DISTANCE = 80
    VICTIM_WORLD_MATCH_DISTANCE = 90.0
    VICTIM_MIN_AREA_RATIO = 0.6
    VICTIM_RECENT_MATCH_WINDOW = 3.0
    VICTIM_TRACK_TTL = 8.0
    VICTIM_PROMOTION_FRAMES = 2
    VICTIM_MIN_TRACK_CONFIDENCE = 0.35
    VICTIM_MIN_TRACK_AREA = 400.0
    VOICE_ENABLED = True
    VOICE_RATE = 170

    # UI Colors (BGR)
    COLOR_BG = (15, 15, 15)
    COLOR_TEXT = (255, 255, 255)
    COLOR_BBOX = (0, 255, 150)
    COLOR_WARNING = (0, 50, 255)
    COLOR_INFO = (255, 200, 0)
