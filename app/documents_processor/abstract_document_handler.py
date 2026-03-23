import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

"""
Abstract class for different type of documents (Word, Pdf and other)
"""

class DocumentHandler(ABC):

    def __init__(self):
        self.config: Optional[Dict] = None
        self.handled_documents: Optional[List] = []
        self.chunked_documents: Optional[List[str]] = []

    @abstractmethod
    def handle_documents(self):
        raise NotImplemented

    def set_config(self, config: Dict):
        self.config = config

    def get_chunked_documents(self) -> List[str]:
        return self.chunked_documents

    def get_handled_documents(self) -> List[str]:
        return self.handled_documents

    def text_chunking(self, document: str) -> List[str]:

        chunk_size = self.config['rag']['chunk_size']
        overlap = self.config['rag']['chunk_overlap']

        chunks: Optional[List] = []

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

    def clean_text(self, text: str) -> str:
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\(\)-]', '', text)
        return text.strip()
