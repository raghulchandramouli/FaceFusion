from facefusion import state_manager
from facefusion.processors.modules import face_swapper

class FaceSwapperInitializer:
    @staticmethod
    def initialize():
        try:
            state_manager.init_item('download_providers', ['huggingface', 'github'])
            state_manager.init_item('download_scope', 'full')
            state_manager.init_item('config_path', 'facefusion.ini')
            state_manager.init_item('source_paths', [])
            state_manager.init_item('target_path', None)
            state_manager.init_item('output_path', None)
            
            FaceSwapperInitializer._setup_execution_provider()
            FaceSwapperInitializer._setup_face_detection()
            FaceSwapperInitializer._setup_face_masking()
            FaceSwapperInitializer._setup_face_swapper()
            FaceSwapperInitializer._setup_output_settings()
            
            if not face_swapper.pre_check():
                return False
            
            _ = face_swapper.get_inference_pool()
            print("✅ Inference pool initialized")
            return True
        except Exception as e:
            print(f"❌ Initialization error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    @staticmethod
    def _setup_execution_provider():
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
        state_manager.init_item('execution_thread_count', 4)  # Increased from 1 to 4
    
    @staticmethod
    def _setup_face_detection():
        state_manager.init_item('face_detector_model', 'yolo_face')
        state_manager.init_item('face_detector_angles', [0])
        state_manager.init_item('face_detector_size', '640x640')  # Reduced from 640x640 for speed
        state_manager.init_item('face_detector_score', 0.6)  # Increased threshold for fewer false positives
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
    
    @staticmethod
    def _setup_face_masking():
        state_manager.init_item('face_occluder_model', None)  # Disable for speed
        state_manager.init_item('face_parser_model', None)  # Disable for speed
        state_manager.init_item('face_mask_types', ['box'])  # Only box mask (fastest)
        state_manager.init_item('face_mask_areas', ['skin', 'left-eyebrow', 'right-eyebrow', 'left-eye', 'right-eye', 'nose', 'mouth', 'upper-lip', 'lower-lip'])
        state_manager.init_item('face_mask_regions', ['skin', 'left-eyebrow', 'right-eyebrow', 'left-eye', 'right-eye', 'nose', 'mouth', 'upper-lip', 'lower-lip'])
        state_manager.init_item('face_mask_blur', 0.1)  # Reduced from 0.3 for speed
        state_manager.init_item('face_mask_padding', [0, 0, 0, 0])
    
    @staticmethod
    def _setup_face_swapper():
        state_manager.init_item('face_swapper_model', 'inswapper_128')
        state_manager.init_item('face_swapper_pixel_boost', '256x256')
        state_manager.init_item('face_swapper_weight', 0.5)
        state_manager.init_item('processors', ['face_swapper'])
        state_manager.init_item('video_memory_strategy', 'tolerant')  # Changed from 'moderate' to 'tolerant'
        state_manager.init_item('system_memory_limit', 0)
    
    @staticmethod
    def _setup_output_settings():
        state_manager.init_item('trim_frame_start', None)
        state_manager.init_item('trim_frame_end', None)
        state_manager.init_item('temp_frame_format', 'jpg')  # Changed from 'png' to 'jpg' for speed
        state_manager.init_item('keep_temp', False)
        state_manager.init_item('output_image_quality', 80)
        state_manager.init_item('output_image_scale', 1.0)
        state_manager.init_item('output_video_fps', None)
        state_manager.init_item('output_video_scale', 1.0)
        state_manager.init_item('output_audio_volume', 100)


