import re
import unittest
from pathlib import Path
from typing import List, Optional

import pdfplumber

from app.logger import LoggerWrapper

logger = LoggerWrapper()


def text_chunking(document: str) -> List[str]:
    chunk_size = 1500
    overlap = 150

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

def clean_text(text: str) -> str:
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\;\:\(\)-]', '', text)
    return text.strip()


class Test(unittest.TestCase):

    def setUp(self):
        pass

    def test_document_processing(self):
        preserve_layout = True
        path = Path(__file__).parent.parent / "documents"
        filename = "icebreaker_eng.pdf"

        if filename.endswith(".pdf"):
            file_path = path / filename
            try:
                logger(file_path)
                with pdfplumber.open(file_path) as pdf:
                    if preserve_layout:
                        full_text = "".join([clean_text(page.extract_text(layout=True)) for page in pdf.pages])
                    else:
                        full_text = "\n".join([page.extract_text().strip() for page in pdf.pages])
                        # full_text = full_text.join([page.extract_tables() for page in pdf.pages])
            except Exception as e:
                logger(f"File in pdf didn't handle [[71]] {file_path}: {e}")

        logger(f"{full_text}")
        logger(f"CHUNKS: {text_chunking(full_text)}")

    if __name__ == '__main__':
        unittest.main()