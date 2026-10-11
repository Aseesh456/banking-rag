from pypdf import PdfReader

def read_pdf(pdf_file):
    text = ""

    pdf = PdfReader(pdf_file)

    for page in pdf.pages:
        text += page.extract_text()

    return text