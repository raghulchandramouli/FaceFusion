import cv2
from wrapper.core.state import app_state

class CameraService:

    @staticmethod
    def initialize():
        if app_state.camera_ready and app_state.camera is not None:
            return True

        try:
            app_state.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            app_state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            app_state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            app_state.camera.set(cv2.CAP_PROP_FPS, 10)
            app_state.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            for _ in range(5):
                app_state.camera.read()

            app_state.camera_ready = True
            print("Camera initialized and ready")
            return True

        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False