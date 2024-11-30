import streamlit as st
from datetime import datetime
import openai
from components.navigation import render_navbar, create_back_button
from utils.pdf_utils import create_diagnostic_report
from config import Config

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
    
    # Sample logit values - replace with actual values from your model
    logit_values = {
        "Wernicke": 0.2,
        "Broca": 0.2,
        "Anomic": 0.5,
        "Control": 0.1
    }
    
    try:
        # Generate report using OpenAI
        prompt = create_report_prompt(user_info, age, logit_values)
        report = generate_openai_report(prompt)
        
        # Display report
        st.write("", report)
        
        # Create and offer PDF download
        pdf_buffer = create_diagnostic_report(user_info, logit_values, report)
        
        st.download_button(
            label="Download PDF",
            data=pdf_buffer,
            file_name="diagnostic_report.pdf",
            mime="application/pdf"
        )
        
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")

def create_report_prompt(user_info: dict, age: int, logit_values: dict) -> str:
    """Create the prompt for OpenAI report generation."""
    current_date = datetime.now().strftime('%Y년 %m월 %d일')
    
    pre_prompt = (
        "한국어로 친절하게 대답해주세요. "
        f"실어증 Logit 값은 다음과 같습니다. "
        f"'Wernicke: {logit_values['Wernicke']}, "
        f"Broca: {logit_values['Broca']}, "
        f"Anomic: {logit_values['Anomic']}, "
        f"Control: {logit_values['Control']}'\n\n"
    )
    
    main_prompt = f"""실어증 유형에 따른 진단 결과 보고서를 작성해주세요...
    [Rest of your prompt template]
    """
    
    return pre_prompt + main_prompt

def generate_openai_report(prompt: str) -> str:
    """Generate report using OpenAI API."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=3000,
        temperature=0.5
    )
    
    return response.choices[0].message.content.strip()