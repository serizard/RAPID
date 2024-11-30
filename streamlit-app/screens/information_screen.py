import streamlit as st
from datetime import datetime
from components.navigation import render_navbar, create_back_button

def render_information_screen():
    render_navbar()
    create_back_button("instruction")
    
    st.markdown('<div class="instruction-header">User Information</div>', 
                unsafe_allow_html=True)

    name = st.text_input("Name")
    
    current_year = datetime.now().year
    birth_year = st.selectbox(
        "Birth Year", 
        list(range(1900, current_year + 1))[::-1]
    )
    
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Next", use_container_width=True):
            st.session_state.user_info = {
                "name": name,
                "birth_year": birth_year,
                "gender": gender
            }
            st.session_state.page = "record"
            st.rerun()