from typing import List, Dict, Optional, TypeVar
from gliner2 import GLiNER2
from app.entity.abstract_entity import AbstractEntity
from app.logger import LoggerWrapper

logger = LoggerWrapper()

"""
Input: 
Output: 
"""

T = TypeVar("T")

class GlinerTwoEntity(AbstractEntity):
    def __init__(self):

        self.gliner: Optional[T] = None
        self.gliner_label: Optional[List[str]] = []
        self.document: Optional[str] = ""
        self.gliner_entities: Optional[List[Dict]] = []

    def set_gliner_model(self):
        try:
            model_ticker = self.config['gliner2']['ticket']
            self.gliner = GLiNER2.from_pretrained(
                model_ticker,
                # device='cuda' if torch.cuda.is_available() else 'cpu',
            )
            logger(f"Loaded {model_ticker} Model")
        except Exception as e:
            logger(f"GLiNER model install ERROR [[81]]: {e}")
            raise

    def set_gliner_label(self, gliner_label: Dict[str: str]):
        self.gliner_label = gliner_label

    def set_text_extraction(self, document: str):
        self.document = ""
        self.document = document

    def get_extract_entities(self) -> List[Dict]:
        return self.gliner_entities

    def extractor_entity(self):
        self.gliner_entities.clear()
        if self.gliner is not None and not self.gliner_label and not self.document:
            gliner_entities = self.gliner.extract_entities(self.document, self.gliner_label, threshold=self.config['gliner2']['threshold'])

            for key, value in gliner_entities['entities'].items():
                value_str = ', '.join(value)
                self.gliner_entities.append({
                    'entity': value_str,
                    'label': key,
                })

        self.gliner_entities.sort(key=lambda x: x['label'], reverse=True)
        logger(f"Entities extracted: {len(self.gliner_entities)} by GLiNER 2 model")