import cv2
import time
import os
import threading
from facefusion.face_analyser import get_many_faces
from facefusion.processors.modules import face_swapper
from wrapper.core.state import app_state


class VideoService:
    
    @staticmethod
    def record_video(frame):
        """Capture all frames for 30 FPS output"""
        if frame is None:
            return frame

        if not getattr(app_state, 'is_recording', False):
            app_state.is_recording = True
            app_state.frames_buffer = []
            app_state.start_time = time.time()
            print("🎬 Recording for 30 FPS output...")

        elapsed = time.time() - app_state.start_time
        
        if elapsed >= 10:
            app_state.is_recording = False
            recorded_path = VideoService._save_recorded_video()
            
            if hasattr(app_state, 'source_face') and app_state.source_face is not None:
                threading.Thread(target=VideoService.process_recorded, daemon=True).start()
            else:
                app_state.processed_video_path = recorded_path
            return frame
        
        # Store every frame
        app_state.frames_buffer.append(frame)
        return frame

    @staticmethod
    def _save_recorded_video():
        """Save as 30 FPS video"""
        if not app_state.frames_buffer:
            return None
        
        timestamp = int(time.time())
        temp_dir = os.path.abspath("temp_data")
        os.makedirs(temp_dir, exist_ok=True)
        video_path = os.path.join(temp_dir, f"recorded_{timestamp}.mp4")
        
        h, w = app_state.frames_buffer[0].shape[:2]
        target_fps = 30
        target_frames = 300  # 30 FPS × 10 seconds
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, target_fps, (w, h))
        
        captured_frames = len(app_state.frames_buffer)
        
        # Create 300 frames by interpolating captured frames
        for i in range(target_frames):
            frame_idx = int((i / target_frames) * captured_frames)
            frame_idx = min(frame_idx, captured_frames - 1)
            
            frame = app_state.frames_buffer[frame_idx]
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr)
        
        out.release()
        app_state.recorded_video_path = video_path
        app_state.recorded_fps = target_fps
        
        print(f"💾 30 FPS Video: 300 frames @ 30 FPS = 10.0s (from {captured_frames} captured)")
        return video_path


    @staticmethod
    def process_recorded():
        """Background face-swap after recording"""
        if not app_state.frames_buffer or app_state.source_face is None:
            return None

        start_time = time.time()
        total_frames = len(app_state.frames_buffer)

        timestamp = int(time.time())
        temp_dir = os.path.abspath("temp_data")
        os.makedirs(temp_dir, exist_ok=True)
        output_path = os.path.join(temp_dir, f"faceswap_{timestamp}.mp4")

        h, w = app_state.frames_buffer[0].shape[:2]
        output_fps = getattr(app_state, 'recorded_fps', 30.0)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (w, h))

        for i, frame in enumerate(app_state.frames_buffer):
            try:
                swapped_frame = frame.copy()
                target_faces = get_many_faces([frame])
                if target_faces:
                    result = face_swapper.swap_face(app_state.source_face, target_faces[0], swapped_frame)
                    if result is not None:
                        swapped_frame = result

                bgr_frame = cv2.cvtColor(swapped_frame, cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)

            except Exception:
                # Fallback to raw frame if processing fails
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)

            # Progress info
            elapsed = time.time() - start_time
            fps_processing = (i + 1) / elapsed if elapsed > 0 else 0
            app_state.processing_progress = {
                'processed': i + 1,
                'total': total_frames,
                'fps': fps_processing,
                'elapsed': elapsed,
                'recorded_fps': output_fps,
                'recorded_frames': total_frames,
                'duration': 10.0
            }

        out.release()
        app_state.processed_video_path = output_path

        print(f"✅ Face-swap completed: {total_frames} frames @ {output_fps:.1f} FPS ≈10 s")
        return output_path

    @staticmethod
    def process_video_file(video_path):
        """Process video file with face swap"""
        if not video_path or not os.path.exists(video_path):
            return None
        
        if app_state.source_face is None:
            return None
        
        # Read video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Output video
        timestamp = int(time.time())
        temp_dir = os.path.abspath("temp_data")
        os.makedirs(temp_dir, exist_ok=True)
        output_path = os.path.join(temp_dir, f"faceswap_{timestamp}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Convert BGR to RGB for face processing
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Apply face swap
                swapped_frame = rgb_frame.copy()
                target_faces = get_many_faces([rgb_frame])
                if target_faces:
                    result = face_swapper.swap_face(app_state.source_face, target_faces[0], swapped_frame)
                    if result is not None:
                        swapped_frame = result
                
                # Convert back to BGR and write
                bgr_frame = cv2.cvtColor(swapped_frame, cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)
                
            except:
                # Fallback to original frame
                out.write(frame)
            
            frame_count += 1
        
        cap.release()
        out.release()
        
        print(f"✅ Face swap completed: {frame_count} frames @ {fps:.1f} FPS")
        return output_path