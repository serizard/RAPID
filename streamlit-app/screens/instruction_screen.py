import streamlit as st
from components.navigation import render_navbar, create_back_button

def render_instruction_screen():
    render_navbar()
    create_back_button("start")
    
    st.markdown('<div class="instruction-header">Test Guidance 📝</div>', 
                unsafe_allow_html=True)
    
    st.markdown(
        '<div class="instruction-text">'
        'This test involves recording a video for aphasia diagnosis. '
        'When you\'re ready, please press the \'Continue Test\' button to begin recording.<br><br>'
        'If you want to read the story of Cinderella before the test, '
        'please click the \'Review Cinderella Story\' button'
        '</div>', 
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Review Cinderella Story", 
                    use_container_width=True,
                    type="secondary"):
            st.session_state.page = "story"
            st.rerun()
        
        st.write("")
        
        if st.button("Continue Test",
                    use_container_width=True,
                    type="primary"):
            st.session_state.page = "information"
            st.rerun()
