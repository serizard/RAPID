
import streamlit as st
from datetime import datetime
import openai
from fpdf import FPDF
import matplotlib.pyplot as plt
import io
from prompts import Prompt1, Prompt2, Prompt3, Prompt4, Prompt5, Prompt6, Prompt7

# Set OpenAI API Key
openai.api_key = "your_openai_api_key"

# PDF Report Class
class ReportPDF(FPDF):
    def header(self):
        self.set_font('NanumGothic', 'B', 15)
        self.cell(80)
        self.cell(30, 10, 'Aphasia Diagnostic Center', 0, 0, 'C')
        self.ln(20)
        self.set_draw_color(0, 80, 180)
        self.set_line_width(0.5)
        self.line(10, 30, 200, 30)

    def footer(self):
        self.set_y(-15)
        self.set_font('NanumGothic', '', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
        self.set_draw_color(0, 80, 180)
        self.line(10, 282, 200, 282)

# Generate Prompts
def define_prompts(user_info, age):
    base_prompts = {
        "Initial Medical Record": Prompt1,
        "Diagnostic Aphasia Analysis": Prompt2,
        "Aphasia Description": Prompt3,
        "Risk Assessment": Prompt4,
        "Communication Improvement Plan": Prompt5,
        "Treatment Recommendations": Prompt6,
        "Medical Summary": Prompt7
    }
    return {
        title: prompt.format(name=user_info["name"], age=age, gender=user_info["gender"]) 
        for title, prompt in base_prompts.items()
    }

# Create PDF Report
def create_pdf_report(sections, user_info):
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.alias_nb_pages()
    pdf.add_font('NanumGothic', '', 'NanumGothic.ttf', uni=True)
    pdf.add_font('NanumGothic', 'B', 'NanumGothicBold.ttf', uni=True)
    
    # Title page
    pdf.add_page()
    pdf.set_font('NanumGothic', 'B', 24)
    pdf.ln(60)
    pdf.cell(0, 20, '실어증 진단 결과 보고서', 0, 1, 'C')
    pdf.ln(20)
    
    # Add sections
    for title, content in sections.items():
        pdf.set_font('NanumGothic', 'B', 14)
        pdf.cell(0, 10, title, 0, 1, 'L')
        pdf.set_font('NanumGothic', '', 12)
        pdf.multi_cell(0, 10, content)
        pdf.ln(10)
    
    pdf_output = io.BytesIO()
    pdf.output(pdf_output, 'F')
    pdf_output.seek(0)
    return pdf_output

# Main Function
def main():
    st.set_page_config(page_title="Aphasia Diagnostic Report Generator", layout="wide")
    
    st.markdown("<h1 style='text-align: center;'>Aphasia Diagnostic Report Generator</h1>", unsafe_allow_html=True)
    with st.sidebar:
        st.header("Patient Information")
        name = st.text_input("Name")
        birth_year = st.number_input("Birth Year", min_value=1900, max_value=datetime.now().year, value=2000)
        gender = st.selectbox("Gender", ["Male", "Female"])
        
        user_info = {
            "name": name,
            "birth_year": birth_year,
            "gender": gender
        }
    
    if st.button("Generate Report"):
        try:
            age = datetime.now().year - user_info["birth_year"]
            prompts = define_prompts(user_info, age)
            
            sections = {}
            for title, prompt in prompts.items():
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a medical professional."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.7
                )
                sections[title] = response.choices[0].message.content.strip()
            
            pdf_output = create_pdf_report(sections, user_info)
            st.download_button(
                label="Download Report",
                data=pdf_output,
                file_name="Aphasia_Report.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
