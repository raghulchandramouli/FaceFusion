import sys
import os

# Add parent directory to path so 'wrapper' module can be found
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from wrapper.ui.gradio_app import create_app

def run_app():
    demo = create_app()
    demo.launch(
        share=True,
        server_name="127.0.0.1",
        server_port=7860,  # Changed port
        show_error=True,
        debug=True
    )

if __name__ == "__main__":
    run_app()
