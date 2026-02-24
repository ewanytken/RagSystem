import torch
from typing import List, Dict, Optional, TypeVar
from gliner import GLiNER

from app.entity.abstract_entity import AbstractEntity
from app.logger import LoggerWrapper

logger = LoggerWrapper

"""
Input: config.yaml with path to ticket gliner model [gliner][ticket] and limit parameter [gliner][threshold]
       documents_extraction - List[str],  gliner_label - List[str] is that need extract from document_extraction  
Output: gliner entities List[Dict]. Example [{'start': 28, 'end': 48, 'text': 'USA', 'label': 'country', 'score': 0.55}, ...]
        sorted by 'score'
"""

T = TypeVar("T")

class GlinerEntity(AbstractEntity):
    def __init__(self):

        self.gliner: Optional[T] = None
        self.gliner_label: Optional[List[str]] = []
        self.documents_extraction: Optional[List[str]] = []
        self.config: Optional[Dict] = None
        self.gliner_entities: Optional[List[Dict]] = []

    def set_gliner_model(self):
        try:
            model_ticker = self.config['models']['ner']
            self.gliner = GLiNER.from_pretrained(
                model_ticker,
                local_files_only=True,
                device='cuda' if torch.cuda.is_available() else 'cpu',
            )
            logger(f"Model loaded {model_ticker} on {'GPU' if torch.cuda.is_available() else 'CPU'}")
        except Exception as e:
            logger(f"GLiNER model install ERROR 80: {e}")
            raise

    def set_config(self, config: Dict):
        self.config = config

    def set_gliner_label(self, gliner_label: List[str]):
        self.gliner_label = gliner_label

    def set_text_extraction(self, documents_extraction: List[str]):
        self.documents_extraction = documents_extraction

    def get_extract_entities(self) -> List[Dict]:
        return self.gliner_entities

    def extractor_entity(self):
        if self.gliner is not None and not self.gliner_label and not self.documents_extraction:
            for document in self.documents_extraction:
                gliner_entities = self.gliner.predict_entities(document, self.gliner_label, threshold=self.config['gliner']['threshold'])

                for entity in gliner_entities:
                    self.gliner_entities.append({
                        'text': entity['text'],
                        'label': entity['label'],
                        'score': entity['score'],
                        'method': 'gliner'
                    })

        self.gliner_entities.sort(key=lambda x: x['score'], reverse=True)
        logger(f"Entities extracted: {len(self.gliner_entities)} by GLiNER model")