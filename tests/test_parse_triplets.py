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
    {"subject": "subject", "predicate": "predicate", "object": "object"},

]
```'''
        triplets = self.parse_json_response(response_by_template)

        logger(f"Parse :{triplets}")
        extracted_relation = self.validate_triplets(triplets)

        logger(f"Extracted relation: {extracted_relation}")

        query_triplet = [{"subject": "АО Селектел", "predicate": "утверждён", "object": "года"},
                         {"subject": "ОТЧЕТ", "predicate": "год", "object": " год"}]

        subject = extracted_relation[-1]['subject']
        relation = extracted_relation[-1]['predicate']
        object = extracted_relation[-1]['object']

        # print(re.search(r'\bАО Селектел\b', str('Совет директоров АО Селектел'), re.IGNORECASE))
        # print(re.search(r'\bутверждён\b', str('утверждён'), re.IGNORECASE))
        # print(re.search(r'года', str('21 апреля 2025 года'), re.IGNORECASE))

        print(subject, relation, object)

        subject_pattern = r'{s}'.format(s=subject)
        relation_pattern = r'{r}'.format(r=relation)
        object_pattern = r'{o}'.format(o=object)

        print(subject_pattern, relation_pattern, object_pattern)

        for u, v, pred in extracted_relation:
            print("u, v, pred: ", u, v, pred)


            print(re.search(subject_pattern, str(u), re.IGNORECASE),
                  re.search(relation_pattern, str(v), re.IGNORECASE),
                  re.search(object_pattern, str(pred), re.IGNORECASE))

            if subject_pattern and not re.search(subject_pattern, str(u), re.IGNORECASE):
                continue

            if relation_pattern and not re.search(relation_pattern, str(pred), re.IGNORECASE):
                continue

            if object_pattern and not re.search(object_pattern, str(v), re.IGNORECASE):
                continue

            logger({'SUBJECT': u,
                'PREDICATE': pred,
                'OBJECT': v
                })
    if __name__ == '__main__':
        unittest.main()