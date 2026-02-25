import json
from typing import Optional, Dict, List, re, Union

import networkx as nx

from app.logger import LoggerWrapper
from app.respondent.interface_respondent import Respondent
from app.utils import Utils

logger = LoggerWrapper

"""
Input: nx.MultiDiGraph() by default     
Output: Relation pattern - ["subject": "Мария Кюри", "predicate": "открыла", "object": "радий"}]  
        Method used use - complex_query(subject_pattern="Marie.*", relation_pattern="dis.*")
"""

class GraphRelationExtractor:
    def __init__(self):
        self.llm_model: Optional[Respondent] = None
        self.graph = nx.MultiDiGraph()

        self.config: Optional[Dict] = None
        self.extracted_relation: Optional[List] = None
        self.entities: Optional[List] = None
        self.documents: Optional[List[str]] = None


    def get_relation_from_documents(self) -> List:
        relation: Optional[List] = []
        if self.llm_model is not None and self.config is not None and self.documents is not None:
            for document in self.documents:
                prompt_template = Utils.load_template(self.config["graph"]["prompts"])
                prompt = prompt_template.format(document=document)
                extracted_relation = self.llm_model.generate(prompt)
                logger(f"Extracted relation: {extracted_relation}")

                start_symbol = "["
                end_symbol = "]"
                start = extracted_relation.find(start_symbol)
                end = extracted_relation.find(end_symbol, start + len(start_symbol))

                if start != -1 and end != -1:
                    extracted_relation = extracted_relation[start+len(start_symbol)-1 : end-1]

                logger(f"Clear substring from extracted relation: {extracted_relation}")
                relation.append(json.loads(extracted_relation))

        return relation

    def set_relation_to_graph(self, extracted_relation: List) -> None:
        for relation in extracted_relation:
            subj, obj, pred = self.extract_triple(relation)
            self.graph.add_edge(subj, obj, label=pred)

    def extract_triple(self, relation):
        subj = relation.get('subject', "None subject")
        obj = relation.get('object', "None object")
        pred = relation.get('predicate', "None predicate")
        return subj, obj, pred

    def search_relation_from_graph(self,
                      subject_pattern: Optional[str] = None,
                      relation_pattern: Optional[str] = None,
                      object_pattern: Optional[str] = None) -> None:

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
                'attributes': data
            })

        if self.config["graph"]["limit"]:
            self.extracted_relation = self.extracted_relation[:self.config["graph"]["limit"]]
        else:
            self.extracted_relation = self.extracted_relation[:5]

    def search_relation_by_entity(self, subject: str) -> None:
        for u, v, data in self.graph.edges(data=True):
            if subject.lower() in str(u).lower():
                self.entities.append({
                    'subject': u,
                    'predicate': data.get('label', ''),
                    'object': v,
                    'attributes': data
                })

    def set_graph(self, graph) -> None:
        self.graph = graph

    def set_documents(self, documents: List[str]) -> None:
        self.documents = documents

    def set_llm_model(self, llm_model: Respondent) -> None:
        self.llm_model = llm_model

    def set_config(self, config: Dict) -> None:
        self.config = config
