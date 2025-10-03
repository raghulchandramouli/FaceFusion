import cv2
import time
import sys
import os
import tempfile
import gradio as gr
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from facefusion.processors.modules import face_swapper
from facefusion import state_manager
from facefusion.face_analyser import get_many_faces, get_one_face

# Global variables
initialized = False
source_face = None
recorded_video_path = None
is_recording = False
camera = None
camera_ready = False

def init_face_swapper():
    global initialized
    if initialized:
        return True
        
    try:
        state_manager.init_item('download_providers', ['huggingface', 'github'])
        state_manager.init_item('download_scope', 'full')
        state_manager.init_item('config_path', 'facefusion.ini')
        state_manager.init_item('source_paths', [])
        state_manager.init_item('target_path', None)
        state_manager.init_item('output_path', None)
        
        # DirectML for Windows GPU acceleration
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            print(f"📋 Available providers: {available}")
            
            if 'DmlExecutionProvider' in available:
                state_manager.init_item('execution_providers', ['directml'])
                print("🚀 Using GPU acceleration (DirectML)")
            elif 'CUDAExecutionProvider' in available:
                state_manager.init_item('execution_providers', ['cpu'])
                print("⚠️ CUDA 12.9 not compatible, using CPU")
            else:
                state_manager.init_item('execution_providers', ['cpu'])
                print("⚠️ Using CPU only")
        except Exception as e:
            state_manager.init_item('execution_providers', ['cpu'])
            print(f"⚠️ Using CPU: {e}")
        
        state_manager.init_item('execution_device_ids', ['0'])
        state_manager.init_item('execution_thread_count', 1)
        state_manager.init_item('face_detector_model', 'yolo_face')
        state_manager.init_item('face_detector_angles', [0])
        state_manager.init_item('face_detector_size', '640x640')
        state_manager.init_item('face_detector_score', 0.5)
        state_manager.init_item('face_landmarker_model', '2dfan4')
        state_manager.init_item('face_landmarker_score', 0.5)
        state_manager.init_item('face_selector_mode', 'reference')
        state_manager.init_item('face_selector_order', 'large-small')
        state_manager.init_item('face_selector_age_start', None)
        state_manager.init_item('face_selector_age_end', None)
        state_manager.init_item('face_selector_gender', None)
        state_manager.init_item('face_selector_race', None)
        state_manager.init_item('reference_face_position', 0)
        state_manager.init_item('reference_face_distance', 0.3)
        state_manager.init_item('reference_frame_number', 0)
        state_manager.init_item('face_occluder_model', 'xseg_1')
        state_manager.init_item('face_parser_model', 'bisenet_resnet_34')
        state_manager.init_item('face_mask_types', ['box'])
        state_manager.init_item('face_mask_areas', ['skin', 'left-eyebrow', 'right-eyebrow', 'left-eye', 'right-eye', 'nose', 'mouth', 'upper-lip', 'lower-lip'])
        state_manager.init_item('face_mask_regions', ['skin', 'left-eyebrow', 'right-eyebrow', 'left-eye', 'right-eye', 'nose', 'mouth', 'upper-lip', 'lower-lip'])
        state_manager.init_item('face_mask_blur', 0.3)
        state_manager.init_item('face_mask_padding', [0, 0, 0, 0])
        state_manager.init_item('face_swapper_model', 'inswapper_128')
        state_manager.init_item('face_swapper_pixel_boost', '128x128')
        state_manager.init_item('face_swapper_weight', 0.5)
        state_manager.init_item('processors', ['face_swapper'])
        state_manager.init_item('video_memory_strategy', 'moderate')
        state_manager.init_item('system_memory_limit', 0)
        state_manager.init_item('trim_frame_start', None)
        state_manager.init_item('trim_frame_end', None)
        state_manager.init_item('temp_frame_format', 'png')
        state_manager.init_item('keep_temp', False)
        state_manager.init_item('output_image_quality', 80)
        state_manager.init_item('output_image_scale', 1.0)
        state_manager.init_item('output_video_fps', None)
        state_manager.init_item('output_video_scale', 1.0)
        state_manager.init_item('output_audio_volume', 100)
        
        if not face_swapper.pre_check():
            return False
        
        _ = face_swapper.get_inference_pool()
        print("✅ Inference pool initialized")
        
        initialized = True
        return True
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def init_camera():
    global camera, camera_ready
    if camera_ready and camera is not None:
        return True
    
    try:
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 10)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        for _ in range(5):
            camera.read()
        
        camera_ready = True
        print("📹 Camera initialized and ready")
        return True
    except Exception as e:
        print(f"❌ Camera init error: {e}")
        return False

def load_source_image(image):
    global source_face
    if image is None:
        return "Please upload a source image"
    
    if not init_face_swapper():
        return "Failed to initialize face swapper"
    
    init_camera()
    
    try:
        source_frame = np.array(image)
        source_faces = get_many_faces([source_frame])
        source_face = get_one_face(source_faces)
        
        if source_face:
            return "✅ Source face loaded! Camera ready to record."
        else:
            return "❌ No face found in source image"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def record_video_with_preview():
    global recorded_video_path, is_recording, source_face, camera
    
    if source_face is None:
        yield None, None, "❌ Please load a source face first"
        return
    
    if not camera_ready or camera is None:
        init_camera()
    
    if not camera.isOpened():
        yield None, None, "❌ Cannot access webcam"
        return
    
    fps = 10
    duration = 10
    
    ret, frame = camera.read()
    if not ret:
        yield None, None, "❌ Cannot read from webcam"
        return

    h, w = frame.shape[:2]
    timestamp = int(time.time())
    temp_dir = tempfile.gettempdir()
    recorded_video_path = os.path.join(temp_dir, f"recorded_{timestamp}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(recorded_video_path, fourcc, fps, (w, h))

    is_recording = True
    start_time = time.time()
    frame_count = 0

    while is_recording:
        ret, frame = camera.read()
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
    is_recording = False
    
    yield None, recorded_video_path, f"✅ Recording complete! {frame_count} frames saved"

def process_recorded_video():
    global recorded_video_path, source_face
    
    if recorded_video_path is None:
        return None, "❌ No recorded video found"
    
    if source_face is None:
        return None, "❌ No source face loaded"
    
    cap = cv2.VideoCapture(recorded_video_path)
    if not cap.isOpened():
        return None, "❌ Cannot open recorded video"
    
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
                    swapped = face_swapper.swap_face(source_face, target_face, temp_frame)
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
    
    status = f"✅ Face swap complete!\n📁 {output_path}\n📊 {processed_frames} frames | 🎭 {successful_frames} swaps"
    return output_path, status

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
    
    load_btn.click(fn=load_source_image, inputs=[source_input], outputs=[status_text])
    record_btn.click(fn=record_video_with_preview, outputs=[live_preview, recorded_video, record_status])
    process_btn.click(fn=process_recorded_video, outputs=[output_video, process_status])

if __name__ == "__main__":
    demo.launch(share=True)




