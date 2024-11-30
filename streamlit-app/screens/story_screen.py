import streamlit as st
from components.navigation import create_back_button
from utils.image_utils import get_image_base64
from config import Config

def render_story_screen():
    st.markdown("<h2 style='text-align: center;'>Cinderella Story 📖</h2>", 
                unsafe_allow_html=True)
    
    if 'story_page' not in st.session_state:
        st.session_state.story_page = 0
    
    current_page = Config.STORY_PAGES[st.session_state.story_page]
    image_path = current_page["image_path"]
    
    st.markdown("""
        <style>
        .storybook-container {
            background-color: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin: 20px auto;
            max-width: 800px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 10, 1])
    with col2:
        image_base64 = get_image_base64(image_path)
        st.markdown(f"""
            <div class="storybook-container">
                <img src="data:image/png;base64,{image_base64}" class="story-image">
                <div class="story-caption">{current_page["caption"]}</div>
                <div class="page-number">Page {st.session_state.story_page + 1} of {len(Config.STORY_PAGES)}</div>
            </div>
        """, unsafe_allow_html=True)
        
        col_prev, col_next = st.columns([1, 1])
        with col_prev:
            if st.session_state.story_page > 0:
                if st.button("◀ Previous Page", use_container_width=True):
                    st.session_state.story_page -= 1
                    st.rerun()
        
        with col_next:
            if st.session_state.story_page < len(Config.STORY_PAGES) - 1:
                if st.button("Next Page ▶", use_container_width=True):
                    st.session_state.story_page += 1
                    st.rerun()
        
        if st.button("Back to the test", type="primary", use_container_width=True):
            st.session_state.page = "instruction"
            st.rerun()