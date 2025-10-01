import cv2
import time
import sys
import os
import tempfile
import gradio as gr
import numpy as np
from PIL import Image
import onnxruntime

# Add facefusion to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from facefusion.processors.modules import face_swapper
from facefusion import state_manager
from facefusion.face_analyser import get_many_faces, get_one_face
from facefusion.vision import read_static_image

# Global variables
initialized = False
source_face = None
recorded_video_path = None

def init_face_swapper():
    """Initialize face swapper with GPU support"""
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
        
        # Force GPU usage with proper provider options
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            
            if 'CUDAExecutionProvider' in available_providers:
                state_manager.init_item('execution_providers', ['CUDAExecutionProvider'])
                print("🚀 Using GPU acceleration (CUDA ONLY)")
            else:
                state_manager.init_item('execution_providers', ['CPUExecutionProvider'])
                print("⚠️ Using CPU (CUDA not available)")
        except:
            state_manager.init_item('execution_providers', ['CPUExecutionProvider'])
            print("⚠️ Using CPU (fallback)")
        
        state_manager.init_item('execution_device_ids', [0])  # Keep as integer
        state_manager.init_item('execution_thread_count', 1)
        
        # Face detector settings
        state_manager.init_item('face_detector_model', 'yolo_face')
        state_manager.init_item('face_detector_angles', [0])  # Keep as integer
        state_manager.init_item('face_detector_size', '640x640')
        state_manager.init_item('face_detector_score', 0.5)
        
        # Face landmarker settings
        state_manager.init_item('face_landmarker_model', '2dfan4')
        state_manager.init_item('face_landmarker_score', 0.5)
        
        # Face selector settings
        state_manager.init_item('face_selector_mode', 'reference')
        state_manager.init_item('face_selector_order', 'large-small')
        state_manager.init_item('face_selector_age_start', None)
        state_manager.init_item('face_selector_age_end', None)
        state_manager.init_item('face_selector_gender', None)
        state_manager.init_item('face_selector_race', None)
        state_manager.init_item('reference_face_position', 0)
        state_manager.init_item('reference_face_distance', 0.3)
        state_manager.init_item('reference_frame_number', 0)
        
        # Face masker settings
        state_manager.init_item('face_occluder_model', 'xseg_1')
        state_manager.init_item('face_parser_model', 'bisenet_resnet_34')
        state_manager.init_item('face_mask_types', ['box'])
        state_manager.init_item('face_mask_areas', ['skin', 'left-eyebrow', 'right-eyebrow', 'left-eye', 'right-eye', 'nose', 'mouth', 'upper-lip', 'lower-lip'])
        state_manager.init_item('face_mask_regions', ['skin', 'left-eyebrow', 'right-eyebrow', 'left-eye', 'right-eye', 'nose', 'mouth', 'upper-lip', 'lower-lip'])
        state_manager.init_item('face_mask_blur', 0.3)
        state_manager.init_item('face_mask_padding', [0, 0, 0, 0])  # Keep as integers
        
        # Face swapper settings
        state_manager.init_item('face_swapper_model', 'inswapper_128')
        state_manager.init_item('face_swapper_pixel_boost', '128x128')
        state_manager.init_item('face_swapper_weight', 0.5)
        state_manager.init_item('processors', ['face_swapper'])

        # Memory settings
        state_manager.init_item('video_memory_strategy', 'moderate')
        state_manager.init_item('system_memory_limit', 0)
        
        # Frame settings
        state_manager.init_item('trim_frame_start', None)
        state_manager.init_item('trim_frame_end', None)
        state_manager.init_item('temp_frame_format', 'png')
        state_manager.init_item('keep_temp', False)
        
        # Output settings
        state_manager.init_item('output_image_quality', 80)
        state_manager.init_item('output_image_scale', 1.0)
        state_manager.init_item('output_video_fps', None)
        state_manager.init_item('output_video_scale', 1.0)
        state_manager.init_item('output_audio_volume', 100)
        
        if not face_swapper.pre_check():
            return False
        
        initialized = True
        return True
    except Exception as e:
        print(f"Initialization error: {e}")
        return False



def load_source_image(image):
    global source_face
    if image is None:
        return "Please upload a source image"
    
    if not init_face_swapper():
        return "Failed to initialize face swapper"
    
    try:
        source_frame = np.array(image)
        source_faces = get_many_faces([source_frame])
        source_face = get_one_face(source_faces)
        
        if source_face:
            return "✅ Source face loaded! Ready to record video."
        else:
            return "❌ No face found in source image"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def record_video():
    """Record 10-second video from webcam with controlled frame rate"""
    global recorded_video_path
    
    if source_face is None:
        return None, "❌ Please load a source face first"
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None, "❌ Cannot access webcam"
    
    # Video settings - your optimized version
    fps = 5
    duration = 10
    total_frames = fps * duration
    
    # Get frame dimensions
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None, "❌ Cannot read from webcam"

    h, w = frame.shape[:2]

    # Create video file
    timestamp = int(time.time())
    temp_dir = tempfile.gettempdir()
    recorded_video_path = os.path.join(temp_dir, f"recorded_{timestamp}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(recorded_video_path, fourcc, fps, (w, h))

    print(f"🎬 Recording 10-second video...")
    start_time = time.time()
    frame_interval = 1.0 / fps
    next_frame_time = start_time
    frame_count = 0

    while frame_count < total_frames:
        now = time.time()
        if now < next_frame_time:
            time.sleep(max(0, next_frame_time - now))
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)
        frame_count += 1
        next_frame_time += frame_interval

        elapsed = now - start_time
        if elapsed >= duration:
            break
    
    cap.release()
    out.release()
    
    status = f"✅ Video recorded!\n📁 {recorded_video_path}\n📊 {frame_count} frames in {duration}s"
    return recorded_video_path, status

def process_recorded_video():
    """Apply face swap to recorded video with debugging"""
    global recorded_video_path, source_face
    
    if recorded_video_path is None:
        return None, "❌ No recorded video found"
    
    if source_face is None:
        return None, "❌ No source face loaded"
    
    # Open recorded video
    cap = cv2.VideoCapture(recorded_video_path)
    if not cap.isOpened():
        return None, "❌ Cannot open recorded video"
    
    # Get video properties with validation
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 5
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video
    timestamp = int(time.time())
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, f"faceswap_{timestamp}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    print(f"🎭 Processing {str(total_frames)} frames...")
    processed_frames = 0
    successful_swaps = 0
    faces_detected = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Default to original frame
        output_frame = frame
        
        try:
            # Convert BGR to RGB for face detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            target_faces = get_many_faces([rgb_frame])
            
            if target_faces:
                faces_detected += 1
                print(f"Frame {str(processed_frames)}: Found {str(len(target_faces))} faces")
                
                # Apply face swap
                temp_frame = rgb_frame.copy()
                for target_face in target_faces:
                    temp_frame = face_swapper.swap_face(source_face, target_face, temp_frame)
                
                # Convert back to BGR for video
                output_frame = cv2.cvtColor(temp_frame, cv2.COLOR_RGB2BGR)
                successful_swaps += 1
            else:
                if processed_frames < 10:  # Log first 10 frames
                    print(f"Frame {str(processed_frames)}: No faces detected")
        
        except Exception as e:
            print(f"Frame {str(processed_frames)} error: {str(e)}")
        
        out.write(output_frame)
        processed_frames += 1
    
    cap.release()
    out.release()
    
    # Ensure all variables are strings in the final status
    status = "✅ Face swap complete!\n📁 " + str(output_path) + "\n📊 Processed " + str(processed_frames) + " frames\n👤 " + str(faces_detected) + " frames with faces detected\n🎭 " + str(successful_swaps) + " successful face swaps"
    return output_path, status



# Gradio UI
with gr.Blocks(title="Record & Face Swap", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎭 Record Video & Apply Face Swap (GPU Optimized)")
    gr.Markdown("1. Upload source face → 2. Record 10s video → 3. Process with face swap")
    
    with gr.Row():
        with gr.Column():
            source_input = gr.Image(type="pil", label="📸 Upload Source Face")
            load_btn = gr.Button("Load Source Face", variant="primary")
            status_text = gr.Textbox(label="Status", interactive=False, lines=2)
            
            gr.Markdown("---")
            record_btn = gr.Button("🎬 Record 10s Video (5 FPS)", variant="secondary", size="lg")
            record_status = gr.Textbox(label="Recording Status", interactive=False, lines=3)
            
            gr.Markdown("---")
            process_btn = gr.Button("🎭 Apply Face Swap", variant="primary", size="lg")
            process_status = gr.Textbox(label="Processing Status", interactive=False, lines=3)
        
        with gr.Column():
            recorded_video = gr.Video(label="📹 Recorded Video")
            output_video = gr.Video(label="🎬 Face Swapped Video")
    
    # Event handlers
    load_btn.click(
        fn=load_source_image,
        inputs=[source_input],
        outputs=[status_text]
    )
    
    record_btn.click(
        fn=record_video,
        outputs=[recorded_video, record_status]
    )
    
    process_btn.click(
        fn=process_recorded_video,
        outputs=[output_video, process_status]
    )

if __name__ == "__main__":
    demo.launch(share=True)
