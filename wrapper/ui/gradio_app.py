import gradio as gr
import os
import time
import cv2
import numpy as np
from PIL import Image
from wrapper.services.face_service import FaceService
from wrapper.services.video_service import VideoService
from wrapper.services.request_service import req_service
from wrapper.core.state import app_state

# Global state
app_state.is_deepfake = False
app_state.live_recording = False
app_state.live_frames = []
app_state.live_uploaded = False

def load_preloaded_faces():
    """Get preloaded face images"""
    images_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Images')
    faces = []
    if os.path.exists(images_dir):
        for filename in sorted(os.listdir(images_dir)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                faces.append(os.path.join(images_dir, filename))
    return faces

def select_face(evt: gr.SelectData):
    """Load selected face from gallery"""
    faces = load_preloaded_faces()
    if evt.index < len(faces):
        img = Image.open(faces[evt.index])
        FaceService.load_source_face(img)
        return "✅ Face loaded"
    return "❌ Failed to load face"

def upload_face(image):
    """Load uploaded face"""
    if image:
        FaceService.load_source_face(image)
        return "✅ Face uploaded"
    return ""

def clear_face():
    """Clear selected face"""
    app_state.source_face = None
    app_state.is_deepfake = False
    return "✅ Face cleared"

def process_faceswap():
    """Apply face swap - marks as deepfake"""
    if not hasattr(app_state, 'recorded_video_path') or not app_state.recorded_video_path:
        return "❌ No video to process", None
    
    if not hasattr(app_state, 'source_face') or not app_state.source_face:
        return "❌ No face selected", None
    
    processed_video = VideoService.process_video_file(app_state.recorded_video_path)
    if processed_video:
        app_state.processed_video_path = processed_video
        app_state.is_deepfake = True
        return "✅ Face swap completed", processed_video
    
    return "❌ Processing failed", None

def upload_to_s3():
    """Upload video with deepfake flag"""
    video_path = getattr(app_state, 'processed_video_path', None) or getattr(app_state, 'recorded_video_path', None)
    
    if not video_path:
        return "❌ No video to upload"
    
    success = req_service(video_path, app_state.is_deepfake)
    status = "DeepFake" if app_state.is_deepfake else "Real"
    return f"✅ Uploaded as {status}" if success else "❌ Upload failed"

def save_recorded_video(video):
    """Save recorded video"""
    if not video:
        return "❌ No video recorded"
    
    temp_dir = os.path.abspath("temp_data")
    os.makedirs(temp_dir, exist_ok=True)
    
    timestamp = int(time.time())
    video_path = os.path.join(temp_dir, f"recorded_{timestamp}.mp4")
    
    # Copy video file
    import shutil
    shutil.copy2(video, video_path)
    
    app_state.recorded_video_path = video_path
    app_state.is_deepfake = False  # Raw video is not deepfake
    
    return "✅ Video saved"

def start_live_recording():
    """Start live recording with 8-second auto-cutoff"""
    app_state.live_recording = True
    app_state.live_frames = []
    app_state.live_start_time = time.time()
    app_state.recording_duration = 8  # 8 seconds
    app_state.live_uploaded = False
    return "🔴 Recording started... (8s auto-stop)"


def stop_live_recording():
    """Stop live recording and save video"""
    if not app_state.live_recording:
        return "❌ No recording in progress"
    
    app_state.live_recording = False
    
    if not app_state.live_frames:
        return "❌ No frames recorded"
    
    # Save video
    temp_dir = os.path.abspath("temp_data")
    os.makedirs(temp_dir, exist_ok=True)
    
    timestamp = int(time.time())
    video_path = os.path.join(temp_dir, f"live_recorded_{timestamp}.mp4")
    
    h, w = app_state.live_frames[0].shape[:2]
    elapsed = time.time() - app_state.live_start_time
    fps = len(app_state.live_frames) / elapsed if elapsed > 0 else 15
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    
    for frame in app_state.live_frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr)
    
    out.release()
    
    app_state.recorded_video_path = video_path
    duration = len(app_state.live_frames) / fps
    
    return f"✅ Recording saved ({duration:.1f}s)"


def live_faceswap(frame):
    """Live face swap processing with recording, auto-cutoff and auto-upload"""
    if frame is None:
        return frame
    
    swapped_frame = frame
    
    # Apply face swap if source face is available
    if hasattr(app_state, 'source_face') and app_state.source_face:
        try:
            from facefusion.face_analyser import get_many_faces
            from facefusion.processors.modules import face_swapper
            
            faces = get_many_faces([frame])
            if faces:
                result = face_swapper.swap_face(app_state.source_face, faces[0], frame.copy())
                if result is not None:
                    swapped_frame = result
                    app_state.is_deepfake = True  # Live swap is deepfake
        except Exception as e:
            print(f"Live swap error: {e}")
    
    # Record frame if recording is active
    if getattr(app_state, 'live_recording', False):
        app_state.live_frames.append(swapped_frame.copy())
        
        # Elapsed time in current recording
        elapsed = time.time() - app_state.live_start_time

        # One-time auto-upload at ~6 seconds
        if not getattr(app_state, 'live_uploaded', False) and elapsed >= 6:
            try:
                temp_dir = os.path.abspath("temp_data")
                os.makedirs(temp_dir, exist_ok=True)

                timestamp = int(time.time())
                partial_path = os.path.join(temp_dir, f"live_recorded_{timestamp}_6s.mp4")

                h, w = app_state.live_frames[0].shape[:2]
                fps_partial = len(app_state.live_frames) / elapsed if elapsed > 0 else 15

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_partial = cv2.VideoWriter(partial_path, fourcc, fps_partial, (w, h))
                for f in app_state.live_frames:
                    bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                    out_partial.write(bgr)
                out_partial.release()

                # Upload the partial 6s clip
                success_partial = req_service(partial_path)
                print(f"☁️ Auto-upload at 6s: {'success' if success_partial else 'failed'} -> {partial_path}")
                app_state.live_uploaded = True
            except Exception as e:
                print(f"Auto-upload error: {e}")
                app_state.live_uploaded = True  # prevent repeated attempts

        # Auto-stop after configured duration (8 seconds)
        if elapsed >= app_state.recording_duration:
            app_state.live_recording = False
            
            # Save and upload video automatically
            temp_dir = os.path.abspath("temp_data")
            os.makedirs(temp_dir, exist_ok=True)
            
            timestamp = int(time.time())
            video_path = os.path.join(temp_dir, f"live_recorded_{timestamp}.mp4")
            
            h, w = app_state.live_frames[0].shape[:2]
            fps = len(app_state.live_frames) / elapsed if elapsed > 0 else 15
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
            
            for f in app_state.live_frames:
                bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                out.write(bgr)
            
            out.release()
            
            app_state.recorded_video_path = video_path
            
            # Auto-upload final clip to S3
            success = req_service(video_path)
            status = "DeepFake" if app_state.is_deepfake else "Real"
            print(f"✅ Auto-uploaded as {status}" if success else "❌ Auto-upload failed")
    
    return swapped_frame



def create_app():
    with gr.Blocks(title="FaceFusion", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎭 FaceFusion")
        
        with gr.Tabs():
            # Tab 1: Video Recording
            with gr.Tab("📹 Video Recording"):
                with gr.Row():
                    # Sidebar
                    with gr.Column(scale=1):
                        gr.Markdown("### Face Selection")
                        
                        # Face gallery
                        faces = load_preloaded_faces()
                        if faces:
                            face_gallery = gr.Gallery(
                                value=faces,
                                columns=2,
                                height=200,
                                show_label=False
                            )
                        else:
                            face_gallery = gr.Markdown("No faces available")
                        
                        # Upload face
                        face_upload = gr.Image(
                            type="pil",
                            sources=["upload"],
                            label="Upload Face"
                        )
                        
                        clear_btn = gr.Button("Clear Face", variant="secondary")
                        face_status = gr.Textbox(label="Status", interactive=False)
                    
                    # Main area
                    with gr.Column(scale=2):
                        # Video recording
                        recorded_video = gr.Video(
                            sources=["webcam"],
                            label="Record Video"
                        )
                        
                        # Output video
                        output_video = gr.Video(label="Processed Video")
                        
                        # Buttons
                        with gr.Row():
                            faceswap_btn = gr.Button("🎭 Apply Face Swap", variant="primary")
                            upload_btn = gr.Button("☁️ Upload Video", variant="secondary")
                        
                        upload_status = gr.Textbox(label="Upload Status", interactive=False)
                
                # Event handlers
                if faces:
                    face_gallery.select(select_face, outputs=[face_status])
                face_upload.change(upload_face, inputs=[face_upload], outputs=[face_status])
                clear_btn.click(clear_face, outputs=[face_status])
                recorded_video.change(save_recorded_video, inputs=[recorded_video], outputs=[face_status])
                faceswap_btn.click(process_faceswap, outputs=[upload_status, output_video])
                upload_btn.click(upload_to_s3, outputs=[upload_status])
            
            # Tab 2: Live Face Swap
            with gr.Tab("🔴 Live Face Swap"):
                with gr.Row():
                    # Sidebar
                    with gr.Column(scale=1):
                        gr.Markdown("### Face Selection")
                        
                        # Face gallery
                        if faces:
                            face_gallery2 = gr.Gallery(
                                value=faces,
                                columns=2,
                                height=200,
                                show_label=False
                            )
                        else:
                            face_gallery2 = gr.Markdown("No faces available")
                        
                        # Upload face
                        face_upload2 = gr.Image(
                            type="pil",
                            sources=["upload"],
                            label="Upload Face"
                        )
                        
                        clear_btn2 = gr.Button("Clear Face", variant="secondary")
                        face_status2 = gr.Textbox(label="Status", interactive=False)
                    
                    # Main area
                    with gr.Column(scale=2):
                        # Live stream
                        live_input = gr.Image(
                            sources=["webcam"],
                            streaming=True,
                            label="Live Input"
                        )
                        
                        live_output = gr.Image(label="Live Output")
                        
                        # Recording controls
                        with gr.Row():
                            start_rec_btn = gr.Button("🔴 Start Recording (10s)", variant="primary")

                        upload_status2 = gr.Textbox(label="Upload Status", interactive=False)

                
                
                # Event handlers
                if faces:
                    face_gallery2.select(select_face, outputs=[face_status2])
                face_upload2.change(upload_face, inputs=[face_upload2], outputs=[face_status2])
                clear_btn2.click(clear_face, outputs=[face_status2])
                live_input.stream(live_faceswap, inputs=[live_input], outputs=[live_output])
                start_rec_btn.click(start_live_recording, outputs=[upload_status2])
                upload_btn.click(upload_to_s3, outputs=[upload_status2])
    
    return app

if __name__ == "__main__":
    demo = create_app()
    demo.launch()
