import streamlit as st
from datetime import datetime
import openai
from components.navigation import render_navbar, create_back_button
from config import Config
from react_components import medical_report


def render_pdf_screen():
    render_navbar()
    create_back_button("result")
    
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
    
    # Logit values for both OpenAI report and React component
    logit_values = st.session_state.get("logit_values", {'Control': 0.6, 'Fluent': 0.2, 'Non-Comprehensive': 0.1, 'Non-Fluent': 0.1})
    prediction = st.session_state.get("prediction", "Control")


    try:
        # React 컴포넌트를 위한 데이터 준비
        patient_info = {
            "name": user_info["name"],
            "birthYear": user_info["birth_year"],
            "gender": user_info["gender"],
            "prediction": prediction,
            "logit_values": logit_values,
            "diagnosisDate": datetime.now().strftime('%Y-%m-%d')
        }
        
        # React 컴포넌트 렌더링
        medical_report(patient_info)
        
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")