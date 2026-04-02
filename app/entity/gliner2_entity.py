from typing import List, Dict, Optional, TypeVar

import torch
from gliner2 import GLiNER2
from app.entity.abstract_entity import AbstractEntity
from app.logger import LoggerWrapper
from app.utils import Utils

logger = LoggerWrapper()

"""
Input: 
Output: 
"""

T = TypeVar("T")

class GlinerTwoEntity(AbstractEntity):
    def __init__(self):

        self.gliner: Optional[T] = None
        self.gliner_label: Optional[List[Dict]] = []
        self.document: Optional[str] = ""
        self.gliner_entities: Optional[List[Dict]] = []

    def __repr__(self):
        return f"Gliner 2 Component"

    def set_gliner_model(self):
        try:
            model_ticker = self.config['gliner2']['ticket']
            self.gliner = GLiNER2.from_pretrained(
                model_ticker,
            )
            self.gliner.to(Utils.get_gpu_id(self.config['gpu']['memory_reserved']) if torch.cuda.is_available() else "cpu")

            logger(f"Loaded {model_ticker} Model on {'GPU' if torch.cuda.is_available() else 'CPU'}")
        except Exception as e:
            logger(f"GLiNER model install ERROR [[81]]: {e}")
            raise

    def set_gliner_label(self, gliner_label: Dict):
        self.gliner_label = gliner_label

    def set_text_extraction(self, document: str):
        self.document = ""
        self.document = document

    def get_extract_entities(self) -> List[Dict]:
        return self.gliner_entities

    def extractor_entity(self, include_confidence: bool = False):
        try:
            self.gliner_entities.clear()
            if self.gliner and self.gliner_label and self.document:
                entities = self.gliner.extract_entities(self.document, self.gliner_label, threshold=self.config['gliner2']['threshold'])
                if not include_confidence:
                    for key, value in entities['entities'].items():
                        value_str = ', '.join(value)
                        self.gliner_entities.append({
                            'entity': value_str,
                            'label': key,
                            'score': self.config['gliner2']['threshold'],
                        })
                else:
                    for key, value in entities['entities'].items():
                        for val in value:
                            self.gliner_entities.append({
                                'label': key,
                                'entity': val['text'],
                                'score': val['confidence'],
                            })

            self.gliner_entities.sort(key=lambda x: x['label'], reverse=True)
            logger(f"Entities extracted: {len(self.gliner_entities)} by GLiNER 2 model")
        except Exception as e:
            logger(f"GlinerTwo extract entities failed: {e}")