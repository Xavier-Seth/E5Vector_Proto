import os
import re
import pytesseract
import shutil
from pdf2image import convert_from_path
from PIL import Image
from docx import Document as DocxDocument
import pandas as pd
from pathlib import Path

class OcrService:
    def __init__(self, tesseract_cmd='tesseract', pdftoppm_cmd='pdftoppm'):
        self.tesseract_cmd = tesseract_cmd
        self.pdftoppm_cmd = pdftoppm_cmd
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

    def extract_text(self, file_path):
        ext = Path(file_path).suffix.lower()
        if ext in ['.png', '.jpg', '.jpeg']:
            return self._extract_from_image(file_path)
        elif ext == '.pdf':
            return self._extract_from_pdf(file_path)
        elif ext == '.docx':
            return self._extract_from_docx(file_path)
        elif ext in ['.xls', '.xlsx']:
            return self._extract_from_excel(file_path)
        elif ext == '.txt':
            return self._extract_from_txt(file_path)
        else:
            return ''

    def _extract_from_image(self, image_path):
        return pytesseract.image_to_string(Image.open(image_path), lang='eng')

    def _extract_from_pdf(self, pdf_path):
        text = ''
        try:
            images = convert_from_path(pdf_path)
            for img in images:
                text += pytesseract.image_to_string(img, lang='eng') + '\n'
        except Exception as e:
            print(f"Error reading PDF: {e}")
        return text.strip()

    def _extract_from_docx(self, docx_path):
        try:
            doc = DocxDocument(docx_path)

            # Extract paragraphs
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

            # Extract tables
            tables = []
            for table in doc.tables:
                for row in table.rows:
                    row_text = ' | '.join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        tables.append(row_text)

            return '\n'.join(paragraphs + tables)

        except Exception as e:
            print(f"Error reading DOCX: {e}")
            return ''

    def _extract_from_excel(self, excel_path):
        try:
            df = pd.read_excel(excel_path, sheet_name=None)
            text = ''
            for sheet_name, sheet in df.items():
                text += ' '.join(sheet.astype(str).fillna('').values.flatten()) + ' '
            return text.strip()
        except Exception as e:
            print(f"Error reading Excel: {e}")
            return ''

    def _extract_from_txt(self, txt_path):
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading TXT: {e}")
            return ''
