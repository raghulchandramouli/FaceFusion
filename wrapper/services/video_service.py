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
            yield None, None
            return

        if not app_state.camera_ready or app_state.camera is None:
            CameraService.initialize()

        if not app_state.camera.isOpened():
            yield None, None
            return

        fps = 10  # Reduced to 10 FPS for realistic timing
        duration = 10
        frame_interval = 1.0 / fps
        
        ret, frame = app_state.camera.read()
        if not ret:
            yield None, None
            return

        h, w = frame.shape[:2]
        timestamp = int(time.time())
        temp_dir = tempfile.gettempdir()
        app_state.recorded_video_path = os.path.join(temp_dir, f"recorded_{timestamp}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(app_state.recorded_video_path, fourcc, fps, (w, h))

        app_state.is_recording = True
        start_time = time.time()
        next_frame_time = start_time
        frame_count = 0

        while app_state.is_recording:
            current_time = time.time()
            elapsed = current_time - start_time
            
            if elapsed >= duration:
                break
            
            # Wait until it's time for the next frame
            if current_time >= next_frame_time:
                ret, frame = app_state.camera.read()
                if not ret:
                    break

                out.write(frame)
                frame_count += 1
                next_frame_time += frame_interval

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield rgb_frame, None
            else:
                time.sleep(0.001)  # Small sleep to prevent CPU spinning

        out.release()
        app_state.is_recording = False

        yield None, app_state.recorded_video_path

    @staticmethod
    def stop_recording():
        """Stop Recording Immediately"""
        app_state.is_recording = False

    @staticmethod
    def record_with_live_faceswap():
        """Record video with real-time face swap preview"""
        if app_state.source_face is None:
            yield None, None
            return

        if not app_state.camera_ready or app_state.camera is None:
            CameraService.initialize()

        if not app_state.camera.isOpened():
            yield None, None
            return

        fps = 10  # Reduced to 10 FPS for realistic timing
        duration = 10
        frame_interval = 1.0 / fps
        
        ret, frame = app_state.camera.read()
        if not ret:
            yield None, None
            return

        h, w = frame.shape[:2]
        timestamp = int(time.time())
        temp_dir = tempfile.gettempdir()
        app_state.recorded_video_path = os.path.join(temp_dir, f"live_faceswap_{timestamp}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(app_state.recorded_video_path, fourcc, fps, (w, h))

        app_state.is_recording = True
        start_time = time.time()
        next_frame_time = start_time
        frame_count = 0
        swap_count = 0

        while app_state.is_recording:
            current_time = time.time()
            elapsed = current_time - start_time
            
            if elapsed >= duration:
                break
            
            # Wait until it's time for the next frame
            if current_time >= next_frame_time:
                ret, frame = app_state.camera.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                swapped_frame = rgb_frame.copy()
                
                try:
                    target_faces = get_many_faces([rgb_frame])
                    if target_faces:
                        for target_face in target_faces:
                            result = face_swapper.swap_face(app_state.source_face, target_face, swapped_frame)
                            if result is not None:
                                swapped_frame = result
                        swap_count += 1
                except Exception as e:
                    print(f"Frame {frame_count} swap error: {e}")

                bgr_frame = cv2.cvtColor(swapped_frame, cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)
                frame_count += 1
                next_frame_time += frame_interval

                yield swapped_frame, None
            else:
                time.sleep(0.001)  # Small sleep to prevent CPU spinning

        out.release()
        app_state.is_recording = False

        yield None, app_state.recorded_video_path


    @staticmethod
    def process_recorded():
        if app_state.recorded_video_path is None:
            return None
        
        if app_state.source_face is None:
            return None

        cap = cv2.VideoCapture(app_state.recorded_video_path)
        if not cap.isOpened():
            return None

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 10
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        timestamp = int(time.time())
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"faceswap_{timestamp}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        print(f"🎭 Processing {total_frames} frames...")
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
                print(f"Frame {processed_frames} error: {e}")

            out.write(output_frame)
            processed_frames += 1
        
        cap.release()
        out.release()

        print(f"✅ Complete! {processed_frames} frames | {successful_frames} swaps")
        return output_path
