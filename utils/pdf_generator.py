from fpdf import FPDF
import tempfile
import os

def create_pdf_report(patient_info, prediction, confidence, ai_report, audience_mode, language):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="AI Osteoarthritis Diagnostic Report", ln=True, align="C")
    
    pdf.set_font("Arial", size=12)
    pdf.ln(5)
    pdf.cell(200, 10, txt=f"Report Mode: {audience_mode} | Language: {language}", ln=True, align="C")
    pdf.ln(10)
    
    # Patient Info
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt="Patient Information", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Age: {patient_info['age']} | Gender: {patient_info['gender']}", ln=True)
    pdf.multi_cell(0, 10, txt=f"Symptoms & History: {patient_info['history']}")
    pdf.ln(5)
    
    # Prediction
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt=f"CNN Model Prediction: {prediction} ({confidence:.2f}% Confidence)", ln=True)
    pdf.ln(5)
    
    # AI Report
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt="AI Clinical Analysis:", ln=True)
    pdf.set_font("Arial", size=12)
    
    # Clean text for PDF rendering
    clean_report = ai_report.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=clean_report)
    
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "OA_Detailed_Report.pdf")
    pdf.output(temp_path)
    return temp_path