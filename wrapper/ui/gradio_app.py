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

def load_preloaded_face(evt: gr.SelectData):
    """Load selected pre-defined face"""
    faces = get_preloaded_faces()
    if evt.index < len(faces):
        img_path = faces[evt.index][0]
        return Image.open(img_path)
    return None

def create_app():
    preloaded_faces = get_preloaded_faces()
    
    # Custom CSS for better styling
    custom_css = """
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .face-gallery {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
    }
    .control-panel {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
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
                ### Real-time Face Swap with GPU Acceleration
                """
            )
        
        with gr.Row():
            # Left Sidebar - Face Selection
            with gr.Column(scale=1, min_width=250):
                gr.Markdown("### 🎨 Face Selection")
                
                with gr.Accordion("📂 Pre-loaded Faces", open=True, elem_classes="face-gallery"):
                    if preloaded_faces:
                        gallery = gr.Gallery(
                            value=[img_path for img_path, _ in preloaded_faces],
                            label="Click to select",
                            columns=2,
                            rows=2,
                            height=280,
                            object_fit="cover",
                            show_label=False
                        )
                    else:
                        gr.Markdown("⚠️ No faces found\n\nAdd images to `wrapper/Images/`")
                
                with gr.Accordion("📤 Custom Upload", open=False):
                    custom_upload = gr.Image(type="pil", label="Upload Face", height=200)
                    upload_btn = gr.Button("✓ Use This Face", variant="secondary", size="sm")
                
                gr.Markdown("---")
                gr.Markdown(
                    """
                    **💡 Tips:**
                    - Use clear, front-facing photos
                    - Good lighting improves results
                    - One face per image works best
                    """
                )
            
            # Main Content Area
            with gr.Column(scale=3):
                # Source Face Display
                with gr.Row(elem_classes="control-panel"):
                    with gr.Column(scale=1):
                        source_input = gr.Image(type="pil", label="📸 Selected Source Face", height=200)
                    with gr.Column(scale=2):
                        gr.Markdown("### 🚀 Quick Start")
                        gr.Markdown(
                            """
                            1. **Select a face** from gallery or upload custom
                            2. **Load the face** using button below
                            3. **Choose recording mode** (Record Only or Live Swap)
                            4. **Process** if needed (only for Record Only mode)
                            """
                        )
                        load_btn = gr.Button("🔄 Load Source Face", variant="primary", size="lg")
                        status_text = gr.Textbox(label="Status", interactive=False, lines=2, show_label=False)
                
                gr.Markdown("---")
                
                # Recording Controls
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🎬 Recording Controls")
                        with gr.Row():
                            record_btn = gr.Button("📹 Record Only (30 FPS)", variant="secondary", size="lg", scale=1)
                            live_swap_btn = gr.Button("🎭 Live Face Swap (10 FPS)", variant="primary", size="lg", scale=1)
                        
                        record_status = gr.Textbox(
                            label="Recording Status", 
                            interactive=False, 
                            lines=2,
                            placeholder="Ready to record..."
                        )
                        
                        gr.Markdown("---")
                        
                        gr.Markdown("### ⚙️ Post-Processing")
                        process_btn = gr.Button("🎭 Apply Face Swap to Recording", variant="primary", size="lg")
                        gr.Markdown("*Only use if you chose 'Record Only' mode*", elem_classes="text-muted")
                        process_status = gr.Textbox(label="Processing Status", interactive=False, lines=2)
                
                # Preview and Output
                with gr.Row():
                    with gr.Column(elem_classes="preview-panel"):
                        gr.Markdown("### 📹 Live Preview")
                        live_preview = gr.Image(label="", streaming=True, height=400, show_label=False)
                    
                    with gr.Column(elem_classes="preview-panel"):
                        gr.Markdown("### 🎬 Output Video")
                        recorded_video = gr.Video(label="", height=400, show_label=False)
        
        # Footer
        with gr.Row():
            gr.Markdown(
                """
                ---
                <div style='text-align: center; color: #666;'>
                    <p>⚡ Powered by FaceFusion | 🎮 GPU Accelerated (DirectML) | 🚀 Optimized for RTX 5060 Ti</p>
                </div>
                """,
                elem_classes="footer"
            )
        
        # Event handlers
        if preloaded_faces:
            gallery.select(fn=load_preloaded_face, outputs=[source_input])
        
        upload_btn.click(fn=lambda x: x, inputs=[custom_upload], outputs=[source_input])
        load_btn.click(fn=FaceService.load_source_face, inputs=[source_input], outputs=[status_text])
        record_btn.click(fn=VideoService.record_video_with_preview, outputs=[live_preview, recorded_video, record_status])
        live_swap_btn.click(fn=VideoService.record_with_live_faceswap, outputs=[live_preview, recorded_video, record_status])
        process_btn.click(fn=VideoService.process_recorded, outputs=[recorded_video, process_status])
    
    return demo
