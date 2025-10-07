class AppState:
    def __init__(self):
        self.initialized = False
        self.source_face = None
        self.recorded_video_path = None
        self.is_recording = False
        self.camera = None
        self.camera_ready = False
        self.frames_buffer = []
        self.start_time = None
        self.recording_mode = None
        self.video_writer = None
        self.frame_count = 0
        self.last_frame_time = None
        self.recorded_fps = 10.0

app_state = AppState()
