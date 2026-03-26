import json
import re
import unittest
from typing import Dict, List, Any

from app.logger import LoggerWrapper

logger = LoggerWrapper()

class Test(unittest.TestCase):
    def parse_json_response(self, content: str) -> List[Dict]:
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            content = json_match.group()
        try:
            triplets = json.loads(content)
            if isinstance(triplets, dict) and 'triplets' in triplets:
                triplets = triplets['triplets']
            return triplets if isinstance(triplets, list) else []
        except json.JSONDecodeError:
            content = content.replace("'", '"')
            content = content.replace("“", '"')
            content = content.replace("”", '"')
            content = re.sub(r',\s*}', '}', content)
            content = re.sub(r',\s*]', ']', content)
            try:
                return json.loads(content)
            except Exception as e:
                logger(f"Parse to json format ERROR [[93]]: {e}")


    def validate_triplets(self, triplets: List[Any]) -> List[Dict[str, str]]:
        validated = []

        try:
            for t in triplets:
                if not isinstance(t, dict):
                    continue

                # Ensure all required keys exist
                triplet = {
                    "subject": str(t.get("subject", "")).strip(),
                    "predicate": str(t.get("predicate", "")).strip().lower().replace(" ", "_"),
                    "object": str(t.get("object", "")).strip()
                }

                # Validate content
                if all([triplet["subject"], triplet["predicate"], triplet["object"]]):
                    # Clean predicate
                    triplet["predicate"] = self.normalize_predicate(triplet["predicate"])
                    validated.append(triplet)
        except Exception as e:
            logger(f"Validate triplets Error [[92]]: {e}")

        return validated

    def normalize_predicate(self, predicate: str) -> str:
        predicate = predicate.lower().strip()
        predicate = re.sub(r'\s+', '_', predicate)

        return predicate

    def setUp(self):
        pass

    def test_document_processing(self):
        response_by_template = '''```json
[
	{"subject": "Совет директоров АО Селектел", "predicate": "утверждён", "object": "21 апреля 2025 года"},
	{"subject": "ГОДОВОЙ ОТЧЕТ", "predicate": "по итогам деятельности за", "object": "2024 год"},
	{"subject": "ГОДОВОЙ ОТЧЕТ", "predicate": "г.", "object": "Санкт-Петербург"},
	{"subject": "ГОДОВОЙ ОТЧЕТ", "predicate": "год", "object": "2025 год"},
	{"subject": "Полное фирменное наименование Общества", "predicate": "наименование", "object": "Акционерное общество Селектел"},
]
```'''
        triplets = self.parse_json_response(response_by_template)

        logger(f"Parse :{triplets}")
        extracted_relation = self.validate_triplets(triplets)

        logger(f"Extracted relation: {extracted_relation}")

    if __name__ == '__main__':
        unittest.main()