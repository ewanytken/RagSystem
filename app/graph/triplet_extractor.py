import json
from typing import Optional, Dict, List
import re
import networkx as nx

from app.logger import LoggerWrapper
from app.respondent.abstract_respondent import Respondent
from app.utils import Utils

logger = LoggerWrapper()

"""
Input: nx.MultiDiGraph() by default     
Output: Relation pattern - ["subject": "Мария Кюри", "predicate": "открыла", "object": "радий"}]  
        Method used use - complex_query(subject_pattern="Marie.*", relation_pattern="dis.*")
"""

class TripletExtractor:
    def __init__(self):
        self.llm_model: Optional[Respondent] = None
        self.config: Optional[Dict] = None

        self.graph = nx.MultiDiGraph()

        self.documents: Optional[List[str]] = None
        self.extracted_relation: Optional[List] = []

        self.extracted_query: Optional[List] = []

    def extract_triplets(self, query: str = None) -> None:
        if self.llm_model is not None and self.config is not None and self.documents is not None:
            for document in self.documents:
                extraction_template = Utils.load_template(self.config["graph"]["prompts"])
                prompt = extraction_template.format(document=document)

                extracted_relation = self.llm_model.generate(prompt)
                logger(f"Extracted relation: {extracted_relation}")

                start_symbol = "["
                end_symbol = "]"
                start = extracted_relation.find(start_symbol)
                end = extracted_relation.find(end_symbol, start + len(start_symbol))

                if start != -1 and end != -1:
                    extracted_relation = extracted_relation[start+len(start_symbol)-1 : end-1]

                logger(f"Clear substring from extracted relation: {extracted_relation}")

                if query is None:
                    self.set_relation_to_graph(json.loads(extracted_relation), document)
                else:
                    self.set_relation_from_query(json.loads(extracted_relation))

    def set_relation_to_graph(self, extracted_relation: List, document: str) -> None:
        for relation in extracted_relation:
            subj = relation.get("subject", "None subject")
            obj = relation.get("object", "None object")
            pred = relation.get("predicate", "None predicate")
            self.graph.add_edge(subj, obj, label=pred, document=document)

    def set_relation_from_query(self, extracted_relation: List):
        self.extracted_query = []
        for relation in extracted_relation:
            subj = relation.get("subject", "None subject")
            obj = relation.get("object", "None object")
            pred = relation.get("predicate", "None predicate")
            self.extracted_query.append([subj, obj, pred])

    def search_relation_from_graph(self,
                      subject_pattern: Optional[str] = None,
                      relation_pattern: Optional[str] = None,
                      object_pattern: Optional[str] = None) -> None:
        self.extracted_relation = []
        try:
            for u, v, data in self.graph.edges(data=True):
                if subject_pattern and not re.search(subject_pattern, str(u), re.IGNORECASE):
                    continue

                pred = data.get('label', '')
                if relation_pattern and not re.search(relation_pattern, str(pred), re.IGNORECASE):
                    continue

                if object_pattern and not re.search(object_pattern, str(v), re.IGNORECASE):
                    continue

                self.extracted_relation.append({
                    'subject': u,
                    'predicate': pred,
                    'object': v,
                    'document': data.get('document', {})
                })

            if self.config["graph"]["limit"]:
                self.extracted_relation = self.extracted_relation[:self.config["graph"]["limit"]]
            else:
                self.extracted_relation = self.extracted_relation[:5]
        except Exception as e:
            logger(f"Search triple Error 90: {e}")

    def search_relation_by_subject(self, subject: str) -> None:
        self.extracted_relation = []
        for u, v, data in self.graph.edges(data=True):
            if subject.lower() in str(u).lower():
                self.extracted_relation.append({
                    'subject': u,
                    'predicate': data.get('label', ''),
                    'object': v,
                    'document': data.get('document', {})
                })

    def set_documents(self, documents: List[str]) -> None:
        self.documents = documents

    def set_llm_model(self, llm_model: Respondent) -> None:
        self.llm_model = llm_model

    def set_graph(self, graph) -> None:
        self.graph = graph

    def set_config(self, config: Dict) -> None:
        self.config = config

    def get_extracted_relation(self) -> List:
        return self.extracted_relation

    def get_extracted_query(self) -> List:
        return self.extracted_query