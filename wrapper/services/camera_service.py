# Remove server-side camera initialization
# The camera will come from client's browser via Gradio
class CameraService:
    
    @staticmethod
    def initialize():
        # No longer needed - client browser handles camera permission
        return True
