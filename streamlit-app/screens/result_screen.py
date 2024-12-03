import streamlit as st
from components.navigation import render_navbar, create_back_button
from components.ui_components import show_spinner
from utils.image_utils import get_image_base64
from utils.inference_utils import get_inference_result
from config import Config
from scipy.special import softmax

def render_result_screen():
    render_navbar()
    create_back_button("record")
    
    st.markdown('<div class="instruction-header">Complete test</div>', 
                unsafe_allow_html=True)
    
    st.markdown(
        "<p style='text-align: center;'>"
        "The test has ended. Please click the button below to view the results."
        "</p>", 
        unsafe_allow_html=True
    )
    
    image_base64 = get_image_base64(Config.BRAIN_IMAGE_PATH)
    st.markdown(
        f'<div class="image-container">'
        f'<img src="data:image/png;base64,{image_base64}" alt="Brain Image">'
        f'</div>',
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Show Results", use_container_width=True):
            show_spinner()
            result = get_inference_result(st.session_state.video_path)
            labels = ['Control', 'Fluent', 'Non_Comprehensive', 'Non_Fluent']
            st.session_state.prediction = labels[int(result["prediction"])]
            st.session_state.logit_values = dict(zip(labels, softmax(result["logits"])))
            st.session_state.highlight_timestamp = result["highlight_timestamp"]
            st.session_state.all_tokens = result["all_tokens"]
            st.session_state.page = "pdf"
            st.rerun()