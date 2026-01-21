import os
import docx
import pdfplumber
import subprocess


def convert_doc_to_docx(doc_path):
    """
    Converts .doc file to .docx using LibreOffice (soffice).
    Returns converted .docx path if successful, else None.
    """
    try:
        subprocess.run(
            [
                "soffice",
                "--headless",
                "--convert-to",
                "docx",
                doc_path,
                "--outdir",
                os.path.dirname(doc_path)
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return doc_path.replace(".doc", ".docx")
    except Exception as e:
        print(f" DOC conversion failed: {doc_path}")
        return None


def extract_text(file_path):
    """
    Extracts text from .docx, .pdf, and .doc resume files.
    """
    text = ""

    try:
        # Handle legacy .doc files
        if file_path.lower().endswith(".doc"):
            converted_path = convert_doc_to_docx(file_path)
            if converted_path:
                file_path = converted_path
            else:
                return ""

        # DOCX
        if file_path.lower().endswith(".docx"):
            doc = docx.Document(file_path)
            text = " ".join(p.text for p in doc.paragraphs)

        # PDF
        elif file_path.lower().endswith(".pdf"):
            with pdfplumber.open(file_path) as pdf:
                pages = [
                    page.extract_text()
                    for page in pdf.pages
                    if page.extract_text()
                ]
                text = " ".join(pages)

        else:
            print(f" Unsupported file skipped: {file_path}")

    except Exception as e:
        print(f" Error processing {file_path}: {e}")

    return text.strip()

