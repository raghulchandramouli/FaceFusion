import gradio as gr
from wrapper.services.face_service import FaceService
from wrapper.services.video_service import VideoService

def create_app():
    with gr.Blocks(title="Record & Face Swap", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎭 Live Record & Face Swap (GPU Optimized)")
        gr.Markdown("1. Upload source face → 2. Record 10s video (with live preview) → 3. Process face swap")
        
        with gr.Row():
            with gr.Column():
                source_input = gr.Image(type="pil", label="📸 Upload Source Face")
                load_btn = gr.Button("Load Source Face", variant="primary")
                status_text = gr.Textbox(label="Status", interactive=False, lines=2)
                
                gr.Markdown("---")
                record_btn = gr.Button("🎬 Start Recording (10s)", variant="secondary", size="lg")
                record_status = gr.Textbox(label="Recording Status", interactive=False, lines=2)
                
                gr.Markdown("---")
                process_btn = gr.Button("🎭 Apply Face Swap", variant="primary", size="lg")
                process_status = gr.Textbox(label="Processing Status", interactive=False, lines=3)
            
            with gr.Column():
                live_preview = gr.Image(label="📹 Live Preview", streaming=True)
                recorded_video = gr.Video(label="📹 Recorded Video")
                output_video = gr.Video(label="🎬 Face Swapped Video")
        
        load_btn.click(fn=FaceService.load_source_face, inputs=[source_input], outputs=[status_text])
        record_btn.click(fn=VideoService.record_with_preview, outputs=[live_preview, recorded_video, record_status])
        process_btn.click(fn=VideoService.process_recorded, outputs=[output_video, process_status])
    
    return demo
