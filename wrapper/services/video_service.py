import cv2
import time
import tempfile
import os
from facefusion.face_analyser import get_many_faces
from facefusion.processors.modules import face_swapper
from wrapper.core.state import app_state
from wrapper.services.camera_service import CameraService

class VideoService:

    @staticmethod
    def record_video_with_preview():
        if app_state.source_face is None:
            yield None, None, "Please load a source face first" 
            return

        if not app_state.camera_ready or app_state.camera is None:
            CameraService.initialize()

        if not app_state.camera.isOpened():
            yield None, None, "Cannot access webcam"
            return

        fps, duration = 10, 10
        ret, frame = app_state.camera.read()
        if not ret:
            yield None, None, "Cannot read from webcam"
            return

        h, w = frame.shape[:2]
        timestamp = int(time.time())
        temp_dir = tempfile.gettempdir()
        app_state.recorded_video_path = os.path.join(temp_dir, f"recorded_{timestamp}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(app_state.recorded_video_path, fourcc, fps, (w, h))

        app_state.is_recording = True
        start_time = time.time()
        frame_count = 0

        while app_state.is_recording:
            ret, frame = app_state.camera.read()
            if not ret:
                break

            out.write(frame)
            frame_count += 1

            elapsed = time.time() - start_time
            remaining = max(0, duration - elapsed)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            status = f"🔴 Recording... {remaining:.1f}s left | Frame: {frame_count}"

            yield rgb_frame, None, status

            if elapsed >= duration:
                break

        out.release()
        app_state.is_recording = False

        yield None, app_state.recorded_video_path, f"Recording complete! {frame_count} frames saved"
            
    @staticmethod
    def process_recorded():
        if app_state.recorded_video_path is None:
            return None, "❌ No recorded video found"
        
        if app_state.source_face is None:
            return None, "❌ No source face loaded"

        cap = cv2.VideoCapture(app_state.recorded_video_path)
        if not cap.isOpened():
            return None, "❌ Error opening recorded video"

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 10
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        timestamp = int(time.time())
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"processed_{timestamp}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        print(f'Processing {total_frames} frames...')
        processed_frames = 0
        successful_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            output_frame = frame

            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                target_faces = get_many_faces([rgb_frame])

                if target_faces:
                    temp_frame = rgb_frame.copy()
                    for target_face in target_faces:
                        swapped = face_swapper.swap_face(app_state.source_face, 
                                                            target_face, 
                                                            temp_frame)
                                                    
                        if swapped is not None:
                            temp_frame = swapped
                    
                    output_frame = cv2.cvtColor(temp_frame, cv2.COLOR_RGB2BGR)
                    successful_frames += 1

            except Exception as e:
                print(f'Error processing frame: {e}')

            out.write(output_frame)
            processed_frames += 1
        
        cap.release()
        out.release()

        status = f"Face Swap Completes!\n {output_path}\n {processed_frames} frames | {successful_swaps} swaps"