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
    
    with gr.Blocks(title="Record & Face Swap", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎭 Live Record & Face Swap (GPU Optimized)")
        
        with gr.Row():
            # Collapsible Sidebar for pre-loaded faces
            with gr.Column(scale=1, min_width=200):
                with gr.Accordion("📂 Pre-loaded Faces", open=True):
                    if preloaded_faces:
                        gallery = gr.Gallery(
                            value=[img_path for img_path, _ in preloaded_faces],
                            label="Click to select",
                            columns=2,
                            rows=2,
                            height=300,
                            object_fit="cover"
                        )
                    else:
                        gr.Markdown("⚠️ No faces in Images folder")
                
                with gr.Accordion("📤 Upload Custom Face", open=False):
                    custom_upload = gr.Image(type="pil", label="Upload your face")
                    upload_btn = gr.Button("Use This Face", variant="secondary", size="sm")
            
            # Main content
            with gr.Column(scale=3):
                gr.Markdown("**1. Select/Upload source face → 2. Choose recording mode → 3. Process (if needed)**")
                
                with gr.Row():
                    with gr.Column():
                        source_input = gr.Image(type="pil", label="📸 Source Face")
                        load_btn = gr.Button("Load Source Face", variant="primary")
                        status_text = gr.Textbox(label="Status", interactive=False, lines=2)
                        
                        gr.Markdown("---")
                        gr.Markdown("### Recording Mode")
                        with gr.Row():
                            record_btn = gr.Button("🎬 Record Only", variant="secondary")
                            live_swap_btn = gr.Button("🎭 Live Face Swap", variant="primary")
                        record_status = gr.Textbox(label="Recording Status", interactive=False, lines=2)
                        
                        gr.Markdown("---")
                        process_btn = gr.Button("🎭 Apply Face Swap (Post-Process)", variant="primary", size="lg")
                        process_status = gr.Textbox(label="Processing Status", interactive=False, lines=3)
                    
                    with gr.Column():
                        live_preview = gr.Image(label="📹 Live Preview", streaming=True)
                        recorded_video = gr.Video(label="📹 Recorded Video")
                        output_video = gr.Video(label="🎬 Face Swapped Video")
        
        # Event handlers
        if preloaded_faces:
            gallery.select(fn=load_preloaded_face, outputs=[source_input])
        
        upload_btn.click(fn=lambda x: x, inputs=[custom_upload], outputs=[source_input])
        load_btn.click(fn=FaceService.load_source_face, inputs=[source_input], outputs=[status_text])
        record_btn.click(fn=VideoService.record_video_with_preview, outputs=[live_preview, recorded_video, record_status])
        live_swap_btn.click(fn=VideoService.record_with_live_faceswap, outputs=[live_preview, recorded_video, record_status])
        process_btn.click(fn=VideoService.process_recorded, outputs=[output_video, process_status])
    
    return demo
