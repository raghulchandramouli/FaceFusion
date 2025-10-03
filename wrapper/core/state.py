class AppState:
    def __init__(self):
        self.initialized = False
        self.source_face = None
        self.recorded_video_path = None
        self.is_recording = False
        self.camera = None
        self.camera_ready = False

app_state = AppState()