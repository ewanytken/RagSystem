from typing import Optional, List, Dict
import re
from app.entity.abstract_entity import AbstractEntity
from app.logger import LoggerWrapper

logger = LoggerWrapper()

"""
Input: documents_extraction - List[str],  regex_entities - List[str] is that need extract from document_extraction  
Output: regex entities List[Dict]. Example [{'start': 28, 'end': 48, 'text': 'USA', 'label': 'country', 'score': 0.55}, ...]
        'score' is 0.9 always
"""

class RegexEntity(AbstractEntity):

    def __init__(self):
        self.document: Optional[str] = ""
        self.regex_entities: Optional[List[Dict]] = []

    def set_text_extraction(self, document: str):
        self.document = document

    def get_extract_entities(self):
        return self.regex_entities

    def extractor_entity(self):
        time_patterns = [
            r'\b\d{1,2}[:.]\d{2}\b', # Output: ['9:30', '17.45', '12:15']
            r'\b(утреннее|дневное|вечернее|ночное)\s+время\b',
        ]

        date_patterns = [
            r'\b\d{1,2}\s*(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\b',
            r'\b\d{1,2}[./]\d{1,2}[./]\d{4}\b',
        ]

        location_patterns = [
            r'\b(северо-запад|северо-восток|юго-запад|юго-восток|север|юг|запад|восток)\b',
            r'\b(город|поселок|деревня|район)\s+[А-ЯЁ][а-яё]+\b',
        ]

        specific_patterns = [
            r'\b\d+\s*[А-ЯЁ]{2,6}\b',
            r'\b[А-ЯЁ]{2,6}\b',
        ]

        patterns = [
            (time_patterns, "время"),
            (date_patterns, "дата"),
            (location_patterns, "расположение"),
            (specific_patterns, "аббревиатура")
        ]

        for pattern_list, label in patterns:
            for pattern in pattern_list:
                matches = re.finditer(pattern, self.document, re.IGNORECASE)
                for match in matches:
                    text = match.group()
                    if label == "аббревиатура":
                        text = re.sub(r'\d+\s*', '', text.strip())

                    self.regex_entities.append({
                        'text': text,
                        'label': label,
                        'score': 0.9,
                        'method': 'regex'
                    })

        logger(f"Entities extracted: {len(self.regex_entities)} by REGEX patterns")