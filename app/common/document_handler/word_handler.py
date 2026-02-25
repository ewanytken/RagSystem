import os
from typing import Dict, Optional, List

from docx import Document

from app.common.document_handler.abstract_document_handler import DocumentHandler
from app.logger import LoggerWrapper

logger = LoggerWrapper

"""
Input: config.yaml with path to Word documents [paths][documents_dir]
Output: List[document - str]   
"""

class WordHandler(DocumentHandler):
    def __init__(self):
        super().__init__()
        self.config: Optional[Dict] = None
        self.handled_documents: Optional[List] = []
        self.chunked_documents: Optional[List[str]] = []

    def set_config(self, config: Dict):
        self.config = config

    def handle_documents(self) -> None:

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
                    self.handled_documents.append(full_text)
                    self.chunked_documents.extend(*self.text_chunking(full_text))

    def get_chunked_documents(self) -> List[str]:
        return self.chunked_documents

    def get_handled_documents(self) -> List[str]:
        return self.handled_documents

    def text_chunking(self, document: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
        chunks:Optional[List] = []

        sentences = document.split('. ')
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + '. '
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + '. '

        if current_chunk:
            chunks.append(current_chunk.strip())

        if len(chunks) > 1 and overlap > 0:
            overlapped_chunks = []
            for i in range(len(chunks) - 1):
                chunk = chunks[i]
                next_chunk_start = chunks[i + 1][:overlap]
                overlapped_chunks.append(chunk + ' ' + next_chunk_start)
            overlapped_chunks.append(chunks[-1])
            chunks = overlapped_chunks

        return chunks
