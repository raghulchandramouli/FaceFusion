import gradio as gr
import os
import shutil
import time
import cv2
import numpy as np
from PIL import Image
from wrapper.services.face_service import FaceService
from wrapper.services.video_service import VideoService
from wrapper.services.request_service import req_service
from wrapper.core.state import app_state


# ------------------ Utility Functions ------------------ #

def save_recorded_video(video):
    """Save recorded video to temp_data directory"""
    if video is None:
        return None

    temp_dir = os.path.abspath("temp_data")
    os.makedirs(temp_dir, exist_ok=True)

    timestamp = int(time.time())
    video_path = os.path.join(temp_dir, f"recorded_{timestamp}.mp4")
    shutil.copy2(video, video_path)

    app_state.recorded_video_path = video_path
    print(f"💾 Video saved: {video_path}")
    return video_path


def start_live_recording():
    """Start recording live face swap stream"""
    app_state.live_recording = True
    app_state.live_frames = []
    app_state.live_start_time = time.time()
    return "🔴 Recording live stream..."


def stop_live_recording():
    """Stop recording and return live face swap video for live_output"""
    if not getattr(app_state, 'live_recording', False):
        return "❌ No recording in progress", None

    app_state.live_recording = False

    if not getattr(app_state, 'live_frames', []):
        return "❌ No frames recorded", None

    # Save live recorded frames as video
    temp_dir = os.path.abspath("temp_data")
    os.makedirs(temp_dir, exist_ok=True)

    timestamp = int(time.time())
    video_path = os.path.join(temp_dir, f"live_recorded_{timestamp}.mp4")

    h, w = app_state.live_frames[0].shape[:2]
    fps = 15  # Live recording FPS

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    for frame in app_state.live_frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr)

    out.release()
    app_state.live_video_path = video_path

    duration = len(app_state.live_frames) / fps
    print(f"💾 Live video saved: {len(app_state.live_frames)} frames @ {fps} FPS = {duration:.1f}s")

    return f"✅ Live video recorded: {duration:.1f}s", video_path


def upload_live_video_to_s3():
    """Upload live recorded video to S3"""
    if not hasattr(app_state, 'live_video_path') or not app_state.live_video_path:
        return "❌ No live video to upload"

    success = req_service(app_state.live_video_path)
    return "✅ Live video uploaded!" if success else "❌ Upload failed"


def live_face_swap(frame):
    """Real-time face swap on streaming frames with recording"""
    if frame is None:
        return frame

    swapped_frame = frame

    if hasattr(app_state, 'source_face') and app_state.source_face is not None:
        try:
            from facefusion.face_analyser import get_many_faces
            from facefusion.processors.modules import face_swapper

            swapped_frame = frame.copy()
            target_faces = get_many_faces([frame])
            if target_faces:
                result = face_swapper.swap_face(app_state.source_face, target_faces[0], swapped_frame)
                if result is not None:
                    swapped_frame = result
        except Exception as e:
            print("⚠️ Swap failed on frame:", e)

    # Record live frames if recording is active
    if getattr(app_state, 'live_recording', False):
        if not hasattr(app_state, 'live_frames'):
            app_state.live_frames = []
        app_state.live_frames.append(swapped_frame.copy())

    return swapped_frame


def get_preloaded_faces():
    """Load pre-defined face images from Images folder"""
    images_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Images')
    faces = []

    if os.path.exists(images_dir):
        for filename in sorted(os.listdir(images_dir)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(images_dir, filename)
                faces.append((img_path, filename.split('.')[0]))

    return faces


def load_face(evt: gr.SelectData):
    """Load selected face"""
    faces = get_preloaded_faces()
    if evt.index < len(faces):
        img_path = faces[evt.index][0]
        img = Image.open(img_path)
        status = FaceService.load_source_face(img)
        return status
    return "❌ Error loading face"


def load_custom_face(image):
    """Load custom uploaded face"""
    if image is not None:
        status = FaceService.load_source_face(image)
        return status
    return ""


def process_face_swap():
    """Apply face swap to recorded video"""
    if not hasattr(app_state, 'recorded_video_path') or not app_state.recorded_video_path:
        raise gr.Error("❌ No recorded video found. Please record a video first.")

    if not hasattr(app_state, 'source_face') or app_state.source_face is None:
        raise gr.Error("❌ Please choose a source face first!")

    processed_video = VideoService.process_video_file(app_state.recorded_video_path)
    if processed_video:
        app_state.processed_video_path = processed_video
        return "✅ Face swap completed!", processed_video

    raise gr.Error("❌ Face swap processing failed. Please try again.")


def upload_video_to_s3():
    """Upload video to S3"""
    if hasattr(app_state, 'processed_video_path') and app_state.processed_video_path:
        success = req_service(app_state.processed_video_path)
        return "✅ Processed video uploaded!" if success else "❌ Upload failed"
    elif hasattr(app_state, 'recorded_video_path') and app_state.recorded_video_path:
        success = req_service(app_state.recorded_video_path)
        return "✅ Raw video uploaded!" if success else "❌ Upload failed"
    return "❌ No video to upload"


def clear_source_face():
    """Clear the selected source face"""
    app_state.source_face = None
    return "✅ Source face cleared"


def create_face_panel():
    """Create face selection panel"""
    preloaded_faces = get_preloaded_faces()

    with gr.Column(scale=1, min_width=250):
        gr.Markdown("### 🎨 Face Selection")

        if preloaded_faces:
            gallery = gr.Gallery(
                value=[img_path for img_path, _ in preloaded_faces],
                columns=2, rows=2, height=250, object_fit="cover", show_label=False
            )
        else:
            gallery = None
            gr.Markdown("⚠️ No preloaded faces")

        custom_upload = gr.Image(
            type="pil", sources=["upload", "webcam"],
            label="Upload Face", height=150
        )

        clear_btn = gr.Button("🗑️ Clear Face", variant="secondary", size="sm")

        status = gr.Textbox(
            interactive=False, show_label=False,
            placeholder="Select or upload a face..."
        )

    return gallery, custom_upload, clear_btn, status


# ------------------ Main App UI ------------------ #

def create_app():
    with gr.Blocks(title="FaceFusion", theme=gr.themes.Soft()) as demo:

        gr.Markdown("# 🎭 FaceFusion", elem_classes="text-center")

        with gr.Tabs():
            # --- Tab 1: Video Recording Workflow ---
            with gr.Tab("📹 Video Recording"):
                with gr.Row():
                    gallery, upload, clear_btn, status = create_face_panel()

                    with gr.Column(scale=2):
                        with gr.Row():
                            record_video = gr.Video(
                                sources=["webcam"],
                                height=400,
                                show_label=False,
                                label="Record Video (30 FPS)"
                            )
                            output_video = gr.Video(height=400, show_label=False)

                        with gr.Row():
                            faceswap_btn = gr.Button("🎭 Apply Face Swap", variant="secondary")
                            upload_btn = gr.Button("☁️ Upload Video to S3", variant="primary")

                        upload_status = gr.Textbox(
                            interactive=False, show_label=False,
                            placeholder="Upload status..."
                        )

                # Handlers
                if gallery:
                    gallery.select(fn=load_face, outputs=[status])
                upload.change(fn=load_custom_face, inputs=[upload], outputs=[status])
                clear_btn.click(fn=clear_source_face, outputs=[status])

                record_video.change(fn=save_recorded_video, inputs=[record_video], outputs=[output_video])
                faceswap_btn.click(fn=process_face_swap, outputs=[upload_status, output_video])
                upload_btn.click(fn=upload_video_to_s3, outputs=[upload_status])

            # --- Tab 2: Live Face Swap ---
            with gr.Tab("🔴 Live Face Swap"):
                with gr.Row():
                    gallery2, upload2, clear_btn2, status2 = create_face_panel()

                    with gr.Column(scale=2):
                        with gr.Row():
                            live_input = gr.Image(
                                sources=["webcam"],
                                streaming=True,
                                height=400,
                                show_label=False,
                                label="Live Webcam"
                            )
                            live_output = gr.Image(
                                height=400,
                                show_label=False,
                                label="Live Face Swap (real-time)"
                            )

                        with gr.Row():
                            start_record_btn = gr.Button("🔴 Start Recording", variant="secondary")
                            stop_record_btn = gr.Button("⏹️ Stop Recording", variant="secondary")
                            upload_live_btn = gr.Button("☁️ Upload Live Video", variant="primary")

                        live_status = gr.Textbox(
                            interactive=False, show_label=False,
                            placeholder="Live recording status..."
                        )

                # Face selection handlers
                if gallery2:
                    gallery2.select(fn=load_face, outputs=[status2])
                upload2.change(fn=load_custom_face, inputs=[upload2], outputs=[status2])
                clear_btn2.click(fn=clear_source_face, outputs=[status2])

                # ✅ Real-time swap stream
                live_input.stream(
                    fn=live_face_swap,
                    inputs=[live_input],
                    outputs=[live_output],
                    time_limit=300,
                    concurrency_limit=1
                )

                # ✅ Recording control
                start_record_btn.click(
                    fn=start_live_recording,
                    outputs=[live_status]
                )

                stop_record_btn.click(
                    fn=stop_live_recording,
                    outputs=[live_status, live_output]
                )

                upload_live_btn.click(
                    fn=upload_live_video_to_s3,
                    outputs=[live_status]
                )

    return demo


# ------------------ Run App ------------------ #

if __name__ == "__main__":
    demo = create_app()
    demo.launch()
