from fpdf import FPDF
import io
import matplotlib.pyplot as plt
from datetime import datetime

def create_diagnostic_report(user_info: dict, logit_values: dict, report_text: str) -> io.BytesIO:
    """Create a PDF diagnostic report."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Add font and content
    pdf.add_font('NanumGothic', '', str(Config.FONT_PATH), uni=True)
    pdf.set_font('NanumGothic', size=12)
    
    # Add report content
    pdf.multi_cell(0, 10, txt=report_text)
    
    # Create and add graph
    fig, ax = plt.subplots()
    ax.bar(logit_values.keys(), logit_values.values())
    ax.set_title("Logit Value by Type of Aphasia")
    ax.set_ylabel("Logit Value")
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    
    pdf.image(img_buffer, x=10, y=pdf.get_y() + 10, w=100)
    
    # Save PDF to buffer
    pdf_output = io.BytesIO()
    pdf.output(pdf_output, 'F')
    pdf_output.seek(0)
    
    return pdf_output