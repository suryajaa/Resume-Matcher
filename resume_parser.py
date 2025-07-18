# resume_parser.py

import fitz  # PyMuPDF

# Function to extract text from a PDF resume
def extract_text_from_pdf(file_path):
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
    except Exception as e:
        print(f"[!] Error reading PDF: {e}")
    return text.strip()
