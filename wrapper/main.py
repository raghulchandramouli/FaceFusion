import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wrapper.ui.gradio_app import create_app

def run_app():
    demo = create_app()
    demo.launch(
        share=True,
        server_name="127.0.0.1",
        server_port=5000,
        show_error=True,
        debug=True
    )

if __name__ == "__main__":
    run_app()
