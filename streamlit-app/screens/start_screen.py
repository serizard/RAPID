import streamlit as st
from components.navigation import render_navbar
from utils.image_utils import get_image_base64
from config import Config

def render_start_screen():
    """Render the start screen."""
    render_navbar()
    
    st.markdown('<div class="main-header">Aphasia Diagnosis Test 🗣️</div>', 
                unsafe_allow_html=True)
    
    image_base64 = get_image_base64(Config.DIAGNOSTIC_IMAGE_PATH)
    st.markdown(
        f'<div class="image-container"><img src="data:image/png;base64,{image_base64}" alt="Doctor Image"></div>',
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Start test", use_container_width=True):
            st.session_state.page = "instruction"
            st.rerun()