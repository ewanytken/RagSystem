import json
import re
import unittest
from typing import Dict, List

from app.logger import LoggerWrapper

logger = LoggerWrapper()

class Test(unittest.TestCase):
    def extract_triplets_json(self, llm_response: str) -> List[Dict]:
        response = llm_response.strip()
        json_match = re.search(r'(\{.*"triplets".*\})', response, re.DOTALL)

        if json_match:
            try:
                json_str = json_match.group(1)
                # Remove any trailing commas (common LLM mistake)
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)

                data = json.loads(json_str)

                if "triplets" in data and isinstance(data["triplets"], list):
                    valid_triplets = []
                    for triplet in data["triplets"]:
                        if all(k in triplet for k in ["subject", "predicate", "object"]):
                            valid_triplets.append({
                                "subject": str(triplet["subject"]),
                                "predicate": str(triplet["predicate"]),
                                "object": str(triplet["object"])
                            })
                    return valid_triplets

            except json.JSONDecodeError as e:
                logger(f" {e}")

    def setUp(self):
        pass

    def test_document_processing(self):
        content = '''json
            {"triplets": [
                    {"subject": "Methodology", "predicate": "for_testing", "object": "Reliability of Large Language Models"},
                    {"subject": "Methodology", "predicate": "based_on", "object": "Principle of Automated Attacks"},
                    {"subject": "Ivan Alekseevich Utkin", "predicate": "affiliated_with", "object": "A.F. Mozhaisky Military Space Academy"},
                    {"subject": "Anton Mikhailovich Martynov", "predicate": "affiliated_with", "object": "A.F. Mozhaisky Military Space Academy"},
                    {"subject": "paper", "predicate": "proposes", "object": "general approach"},
                    {"subject": "approach", "predicate": "to_studying", "object": "reliability of LLMs"}
            ]}'''
        triplets = self.extract_triplets_json(content)
        logger(triplets)

    if __name__ == '__main__':
        unittest.main()