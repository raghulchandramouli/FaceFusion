import gradio as gr
import os
from PIL import Image
from wrapper.services.face_service import FaceService
from wrapper.services.video_service import VideoService

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

def load_and_process_face(evt: gr.SelectData):
    """Load selected pre-defined face and process it"""
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

def create_app():
    preloaded_faces = get_preloaded_faces()
    
    custom_css = """
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .preview-panel {
        border: 2px solid #667eea;
        border-radius: 10px;
        padding: 10px;
    }
    """
    
    with gr.Blocks(title="FaceFusion Live", theme=gr.themes.Soft(), css=custom_css) as demo:
        # Header
        with gr.Row(elem_classes="main-header"):
            gr.Markdown(
                """
                # 🎭 FaceFusion Live
                ### Real-time Face Swap & DeepFake Detection
                """
            )
        
        with gr.Row():
            # Left Sidebar - Face Selection
            with gr.Column(scale=1, min_width=250):
                gr.Markdown("### 🎨 Face Selection")
                
                with gr.Accordion("📂 Pre-loaded Faces", open=True):
                    if preloaded_faces:
                        gallery = gr.Gallery(
                            value=[img_path for img_path, _ in preloaded_faces],
                            label="Click to select (auto-loads)",
                            columns=2,
                            rows=2,
                            height=280,
                            object_fit="cover",
                            show_label=False
                        )
                    else:
                        gr.Markdown("⚠️ No faces found\n\nAdd images to `wrapper/Images/`")
                
                with gr.Accordion("📤 Upload Custom Face", open=False):
                    custom_upload = gr.Image(
                        type="pil",
                        sources=["upload", "webcam"],
                        label="Upload Face Image",
                        height=200
                    )


                gr.Markdown("---")
                
                status_text = gr.Textbox(label="Status", interactive=False, lines=2, show_label=False, placeholder="Select a face to begin...")
            
            # Main Content Area
            with gr.Column(scale=3):
                gr.Markdown("### 🎬 Recording Controls")
                with gr.Row():
                    record_btn = gr.Button("📹 Record Only", variant="secondary", size="lg", scale=1)
                    live_swap_btn = gr.Button("🎭 Live Face Swap", variant="primary", size="lg", scale=1)
                
                gr.Markdown("---")
                
                # Preview and Output
                with gr.Row():
                    with gr.Column(elem_classes="preview-panel"):
                        gr.Markdown("### 📹 Live Preview")
                        live_preview = gr.Image(label="", streaming=True, height=400, show_label=False)
                        stop_btn = gr.Button("⏹️ Stop Recording", variant="stop", size="lg")
                    
                    with gr.Column(elem_classes="preview-panel"):
                        gr.Markdown("### 🎬 Output Video")
                        recorded_video = gr.Video(label="", height=400, show_label=False)
                        process_btn = gr.Button("🔄 Process Video", variant="secondary", size="lg")
                        gr.Markdown("*Only use if you choose 'Record Only Mode'*")
        
        # Event handlers
        if preloaded_faces:
            gallery.select(fn=load_and_process_face, outputs=[status_text])
        
        custom_upload.change(fn=load_custom_face, inputs=[custom_upload], outputs=[status_text])
        
        record_btn.click(fn=VideoService.record_video_with_preview, outputs=[live_preview, recorded_video])
        live_swap_btn.click(fn=VideoService.record_with_live_faceswap, outputs=[live_preview, recorded_video])
        stop_btn.click(fn=VideoService.stop_recording)
        process_btn.click(fn=VideoService.process_recorded, outputs=[recorded_video])
    
    return demo
