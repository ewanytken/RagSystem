import os
from pathlib import Path
from typing import Optional

import pdfplumber
from docx import Document

from app.documents_processor.abstract_document_handler import DocumentHandler
from app.logger import LoggerWrapper

logger = LoggerWrapper()

"""
Input: config.yaml with path to Word documents [paths][documents_dir]
Output: List[document - str]   
"""

class WordPdfHandler(DocumentHandler):

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f"Word and Pdf Handler Component"

    def handle_documents(self) -> None:

        preserve_layout: bool = True

        path = Path(__file__).parent.parent.parent / self.config['paths']['documents_dir']
        logger(f"Path to document load: {path}")

        full_text: Optional[str] = ""

        for filename in os.listdir(path):
            if filename.endswith(".docx") or filename.endswith(".doc"):
                file_path = path / filename
                try:
                    doc = Document(file_path)
                    full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                except Exception as e:
                    logger(f"File in doc (docx) didn't handle [[70]] {file_path}: {e}")

                if full_text is not None:
                    self.handled_documents.append(full_text)
                    self.chunked_documents.extend(self.text_chunking(full_text))

            if filename.endswith(".pdf"):
                file_path = path / filename
                try:
                    with pdfplumber.open(file_path) as pdf:
                        if preserve_layout:
                            full_text = "\n".join([self.clean_text(page.extract_text(layout=True)) for page in pdf.pages])
                        else:
                            full_text = "\n".join([page.extract_text().strip() for page in pdf.pages])
                except Exception as e:
                    logger(f"File in pdf didn't handle [[71]] {file_path}: {e}")

                if full_text is not None:
                    self.handled_documents.append(full_text)
                    self.chunked_documents.extend(self.text_chunking(full_text))

        logger(f"Number of preprocessing documents: {len(self.handled_documents)} \n"
               f"Number of preprocessing chunks: {len(self.chunked_documents)}")
