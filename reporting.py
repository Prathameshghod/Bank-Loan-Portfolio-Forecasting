import pandas as pd
from fpdf import FPDF

def export_csv(df, filename='export.csv'):
    df.to_csv(filename, index=False)
    return filename

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Bank Loan Portfolio Report', 0, 1, 'C')
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, '', 0, 0, 'C')  # No custom footer

def export_pdf(summary_dict, filename='report.pdf'):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 11)
    for key, value in summary_dict.items():
        pdf.cell(0, 10, f'{key}: {value}', ln=1)
    pdf.output(filename)
    return filename
