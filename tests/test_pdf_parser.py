import json
import re
import unittest
from typing import Dict, List

from app.logger import LoggerWrapper

logger = LoggerWrapper()
import pdfplumber
from typing import Optional, List, Dict
import os


class PDFParserPlumber:

    def __init__(self):
        self.pages = None
        self.file_path = None

    def parse_pdf(self, file_path: str, pages: Optional[List[int]] = None,
                  preserve_layout: bool = True) -> str:
        """
        Parse PDF with preservation of layout

        Args:
            file_path: Path to PDF file
            pages: List of page numbers to extract (None for all pages)
            preserve_layout: Whether to preserve text layout

        Returns:
            Extracted text as string
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        try:
            self.file_path = file_path
            text_parts = []

            with pdfplumber.open(file_path) as pdf:
                page_range = pages if pages else range(len(pdf.pages))

                for page_num in page_range:
                    page = pdf.pages[page_num]

                    if preserve_layout:
                        # Preserve original layout
                        text = page.extract_text(layout=True)
                    else:
                        # Simple text extraction
                        text = page.extract_text()

                    if text and text.strip():
                        text_parts.append(text)

            return "\n\n".join(text_parts)

        except Exception as e:
            raise Exception(f"Failed to parse PDF: {e}")

    def extract_tables(self, file_path: str, page_num: int = 0) -> List[List[List[str]]]:
        """
        Extract tables from PDF
        """
        with pdfplumber.open(file_path) as pdf:
            page = pdf.pages[page_num]
            tables = page.extract_tables()
            return tables


# Usage

class Test(unittest.TestCase):

    def setUp(self):
        pass

    def test_document_processing(self):
        parser = PDFParserPlumber()
        text = parser.parse_pdf("document.pdf", preserve_layout=True)
        print(text)

    if __name__ == '__main__':
        unittest.main()