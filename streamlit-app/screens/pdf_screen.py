
import streamlit as st
from datetime import datetime
import openai
from components.navigation import render_navbar, create_back_button
from config import Config
from components.react_components import medical_report
from utils.video_utils import (
    video_player_with_attention_tracking,
    render_attention_box
)
def render_video_section():
    st.markdown('<div class="instruction-header">🎥 Speech Analysis</div>', 
                unsafe_allow_html=True)
    
    if 'all_tokens' in st.session_state:
        
        try:
            video_path = st.session_state.video_path
            max_score = max(st.session_state.all_tokens['importance'])
            
            # Create warning placeholder
            warning_placeholder = st.empty()
            
            # Run video player
            player_status = video_player_with_attention_tracking(
                video_path,
                tokens_data=st.session_state.all_tokens,
                max_score=max_score,
                width = 640,  
                height = 360  
            )
            
            if player_status:
                current_score = player_status["score"]
                render_attention_box(warning_placeholder, current_score, max_score)
            
            # Bookmarks section
            with st.expander("🔖 북마크된 구간"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if st.button("현재 구간 북마크"):
                        if 'bookmarks' not in st.session_state:
                            st.session_state.bookmarks = []
                        
                        if player_status:
                            st.session_state.bookmarks.append({
                                'time': player_status["time"],
                                'score': player_status["score"]
                            })
                            st.experimental_rerun()
                
                with col2:
                    if 'bookmarks' in st.session_state and st.session_state.bookmarks:
                        for bookmark in sorted(st.session_state.bookmarks, key=lambda x: x['time']):
                            score_color = (
                                "🟢" if bookmark['score'] < max_score/3 else
                                "🟡" if bookmark['score'] < max_score*2/3 else
                                "🔴"
                            )
                            st.write(
                                f"{score_color} {bookmark['time']:.1f}초 "
                                f"(위험도: {bookmark['score']:.2f})"
                            )
            
        except Exception as e:
            st.error(f"Error playing video: {str(e)}")
    else:
        st.warning("Speech analysis data not available.")


def render_pdf_screen():
    render_navbar()
    create_back_button("result")
    
    row1_container = st.container()
    row2_container = st.container()
    
    st.markdown("""
        <style>
        .video-section {
            min-height: 600px;  
            margin-bottom: 2rem;
        }
        .report-section {
            min-height: 800px; 
        }
        </style>
    """, unsafe_allow_html=True)
    
    with row1_container:
        st.markdown('<div class="video-section">', unsafe_allow_html=True)
        render_video_section()
        st.markdown('</div>', unsafe_allow_html=True)
    

    with row2_container:
        st.markdown('<div class="instruction-header">📄 Diagnostic Report</div>', 
                   unsafe_allow_html=True)
        
        user_info = st.session_state.get("user_info", {
            "name": "John Doe",
            "birth_year": 1977,
            "gender": "Male"
        })
        
        current_year = datetime.now().year
        age = current_year - user_info["birth_year"]
        
        openai.api_key = Config.OPENAI_API_KEY
        
        logit_values = st.session_state.get("logit_values", 
                                          {'Control': 0.6, 'Fluent': 0.2, 
                                           'Non-Comprehensive': 0.1, 'Non-Fluent': 0.1})
        prediction = st.session_state.get("prediction", "Control")
        timestamp = st.session_state.get("highlight_timestamp", (80,90))
        tokens = st.session_state.get("all_tokens")

        try:
            patient_info = {
                "name": user_info["name"],
                "birthYear": user_info["birth_year"],
                "gender": user_info["gender"],
                "prediction": prediction,
                "logit_values": logit_values,
                "diagnosisDate": datetime.now().strftime('%Y-%m-%d'),
                "timestamp": timestamp,
                "tokens": tokens
            }
            
            medical_report(patient_info)
            
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
