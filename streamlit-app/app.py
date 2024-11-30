# app.py
import streamlit as st
from pathlib import Path

# Import all screen renderers at once
from screens import (
    render_start_screen,
    render_story_screen,
    render_instruction_screen,
    render_information_screen,
    render_record_screen,
    render_result_screen,
    render_pdf_screen
)
from components.ui_components import load_css

def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "start"
    
    if 'recording_complete' not in st.session_state:
        st.session_state.recording_complete = False
    
    # Load CSS
    css_path = Path(__file__).parent / "app.css"
    load_css(css_path)
    
    # Route to appropriate screen
    screens = {
        "start": render_start_screen,
        "story": render_story_screen,
        "instruction": render_instruction_screen,
        "information": render_information_screen,
        "record": render_record_screen,
        "result": render_result_screen,
        "pdf": render_pdf_screen
    }
    
    current_screen = screens.get(st.session_state.page)
    if current_screen:
        current_screen()
    else:
        st.error("Page not found")

if __name__ == "__main__":
    main()