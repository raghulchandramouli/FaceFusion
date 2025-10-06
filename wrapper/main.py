import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wrapper.ui.gradio_app import create_app

def run_app():
    demo = create_app()
    demo.launch(
        share=False,
        server_name="localhost",
        server_port=7860,
        show_error=True
    )

if __name__ == "__main__":
    run_app()
