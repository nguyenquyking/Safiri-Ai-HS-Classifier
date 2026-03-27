import sys
from pypdf import PdfReader

try:
    reader = PdfReader('safiri_take_home_problem2.pdf')
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    with open('pdf_out.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Success")
except Exception as e:
    print(f"Error: {e}")
