import os
import unittest
from pathlib import Path

from docx import Document

from app.logger import LoggerWrapper
from app.utils import Utils

logger = LoggerWrapper()

class Test(unittest.TestCase):

    def setUp(self):
        self.config = Utils.get_config_file()

    def test_document_processing(self):
        path = Path(__file__).parent.parent / self.config['paths']['documents_dir']
        logger(f"Path to document load {path}")
        for filename in os.listdir(path):
            if filename.endswith(".docx") or filename.endswith(".doc"):
                file_path = path / filename
                try:
                    doc = Document(file_path)
                    logger("\n".join([p.text for p in doc.paragraphs if p.text.strip()]))
                except Exception as e:
                    logger(f"File didn't handle 70 {file_path}: {e}")

if __name__ == '__main__':
    unittest.main()