import numpy as np
from facefusion.face_analyser import get_many_faces, get_one_face
from wrapper.core.state import app_state
from wrapper.core.initializer import FaceSwapperInitializer
from wrapper.services.camera_service import CameraService

class FaceService:

    @staticmethod
    def load_source_face(image):

        if image is None:
            return "Please upload a source image"

        if not app_state.initialized:
            if not FaceSwapperInitializer.initialize():
                return "Failed to initialize face swapper"
            app_state.initialized = True

        CameraService.initialize()

        try:
            source_frame = np.array(image)
            source_faces = get_many_faces([source_frame])
            app_state.source_face = get_one_face(source_faces)

            if app_state.source_face:
                return "Source face loaded! Camera ready to record."
            else:
                return "No face found in source image"

        except Exception as e:
            return f"Error: {str(e)}"

