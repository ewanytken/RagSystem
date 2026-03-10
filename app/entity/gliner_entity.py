from typing import List, Dict, Optional, TypeVar
from gliner import GLiNER
from app.entity.abstract_entity import AbstractEntity
from app.logger import LoggerWrapper

logger = LoggerWrapper()

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
        self.document: Optional[str] = ""
        self.gliner_entities: Optional[List[Dict]] = []

    def set_gliner_model(self):
        try:
            model_ticker = self.config['gliner']['ticket']
            self.gliner = GLiNER.from_pretrained(
                model_ticker,
                local_files_only=self.config['gliner']['local_only'],
                # device='cuda' if torch.cuda.is_available() else 'cpu',
            )
            logger(f"Loaded {model_ticker} Model")
        except Exception as e:
            logger(f"GLiNER model install ERROR [[80]]: {e}")
            raise

    def extractor_entity(self) -> None:
        try:
            self.gliner_entities.clear()
            if self.gliner and self.gliner_label and self.document:
                entities = self.gliner.predict_entities(self.document, self.gliner_label, threshold=self.config['gliner']['threshold'])

                for entity in entities:
                    self.gliner_entities.append({
                        'entity': entity['text'],
                        'label': entity['label'],
                        'score': entity['score'],
                    })

            self.gliner_entities.sort(key=lambda x: x['label'], reverse=True)
            logger(f"Entities extracted: {len(self.gliner_entities)} by GLiNER model")
        except Exception as e:
            logger(f"Gliner extract entities failed [[82]]: {e}")

    def set_gliner_label(self, gliner_label: Dict) -> None:
        self.gliner_label = gliner_label

    def set_text_extraction(self, document: str) -> None:
        self.document = ""
        self.document = document

    def get_extract_entities(self) -> List[Dict]:
        return self.gliner_entities