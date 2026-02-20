from typing import Optional, List, Dict
import re
from app.entity.abstract_entity import AbstractEntity
from app.logger import LoggerWrapper

logger = LoggerWrapper

"""

"""

class GlinerEntity(AbstractEntity):
    def __init__(self):

        self.documents_extraction: Optional[List[str]] = []
        self.config: Optional[Dict] = None
        self.regex_entities: Optional[List[Dict]] = []

    def set_text_extraction(self, text_extraction: List[str]):
        self.documents_extraction = text_extraction

    def get_extract_entities(self):
        return self.regex_entities

    def extractor_entity(self):
#TODO complete method
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

        patterns = [
            (time_patterns, "время"),
            (date_patterns, "дата"),
            (location_patterns, "локация"),
        ]

        for pattern_list, label in patterns:
            for pattern in pattern_list:
                matches = re.finditer(pattern, self.documents_extraction, re.IGNORECASE)
                for match in matches:
                    self.regex_entities.append({
                        'text': match.group(),
                        'label': label,
                        'start': match.start(),
                        'end': match.end(),
                        'score': 0.9,
                        'method': 'regex'
                    })

        # Specific labels
        patterns = [
            r'\b\d+\s*[А-ЯЁ]{2,6}\b',
            r'\b[А-ЯЁ]{2,6}\b',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                label_temp = match.group().strip()
                label = re.sub(r'\d+\s*', '', label_temp)

                if label in self.abbreviation_dict:
                    self.regex_entities.append({
                        'text': label_temp,
                        'label': self.abbreviation_dict[label],
                        'start': match.start(),
                        'end': match.end(),
                        'score': 0.95,
                        'method': 'dictionary'
                    })

        logger(f"Entities extracted: {len(self.regex_entities)} by GLiNER model")