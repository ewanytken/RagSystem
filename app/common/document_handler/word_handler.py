import os
from typing import Dict, Optional, List

from docx import Document

from app.common.document_handler.abstract_document_handler import DocumentHandler
from app.logger import LoggerWrapper

logger = LoggerWrapper

"""
Input: config.yaml with path to Word documents [paths][documents_dir]
Output: List[str] str is word's document 
"""

class WordHandler(DocumentHandler):
    def __init__(self):
        super().__init__()
        self.config: Optional[Dict] = None
        self.handled_document: Optional[List] = []

    def set_config(self, config: Dict):
        self.config = config

    def handle_documents(self):

        logger(f"Path to document load {self.config['paths']['documents_dir']}")

        full_text: Optional[str] = ""

        for filename in os.listdir(self.config['paths']['documents_dir']):
            if filename.endswith(".docx") or filename.endswith(".doc"):
                file_path = os.path.join(self.config['paths']['documents_dir'], filename)

                try:
                    doc = Document(file_path)
                    full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                except Exception as e:
                    logger(f"File didn't handle 70 {file_path}: {e}")

                if full_text is not None:
                    self.handled_document.append(full_text)

    def get_handled_documents(self) -> List[str]:
        return self.handled_document