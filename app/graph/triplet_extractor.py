import json
import re
from pathlib import Path
from typing import Optional, Dict, List, Any

import networkx as nx

from app.logger import LoggerWrapper
from app.respondent.abstract_respondent import Respondent
from app.utils import Utils

logger = LoggerWrapper()

"""
Input: nx.MultiDiGraph() by default     
Output: Relation pattern - ["subject": "Мария Кюри", "predicate": "открыла", "object": "радий"}]  
        Method used use - search_relation_from_graph(subject_pattern="Marie.*", relation_pattern="dis.*")
"""

class TripletExtractor:
    def __init__(self):
        self.llm_model: Optional[Respondent] = None
        self.config: Optional[Dict] = None
        self.documents: Optional[List[str]] = None

        self.graph = nx.MultiDiGraph()

        self.triplets_from_graph: Optional[List] = []
        self.triplets_from_query: Optional[List] = []

    def __repr__(self):
        return f"Graph Triplet Component"

    def extract_triplets(self, query: str = None) -> None:
        if self.llm_model and self.config and self.documents:
            try:
                if query is None:
                    if Path("triplet_graph.graphml").is_file():
                        self.graph = nx.read_graphml("triplet_graph.graphml")
                        return

                    for document in self.documents:
                        document = self.clean_text(document)
                        extraction_template = Utils.load_template(self.config['graph']['extractor_prompt'])
                        prompt = extraction_template.format(document=document)

                        response_by_template = self.llm_model.generate(prompt)
                        logger(f"Response after extracted for document: {response_by_template}")

                        triplets = self.parse_json_response(response_by_template)
                        extracted_relation = self.validate_triplets(triplets)

                        logger(f"Extracted relation from document: {extracted_relation}")

                        self.set_relation_to_graph(extracted_relation, document)
                        self.set_relation_to_graph(self.create_inverse_relationships(extracted_relation), document)
                    nx.write_graphml(self.graph, "triplet_graph.graphml")
                else:
                    query = self.clean_text(query)
                    extraction_template = Utils.load_template(self.config['graph']['extractor_prompt'])
                    prompt = extraction_template.format(document=query)

                    response_by_template = self.llm_model.generate(prompt)
                    logger(f"Response after extracted for query: {response_by_template}")

                    triplets = self.parse_json_response(response_by_template)
                    extracted_relation = self.validate_triplets(triplets)

                    logger(f"Extracted relation from query: {extracted_relation}")
                    self.set_relation_from_query(extracted_relation)
            except Exception as e:
                logger(f"Triplet extraction Exception [[91]]: {e}")

        else:
            logger(f"LLM extractor {type(self.llm_model)}, config {type(self.config)} or documents {type(self.documents)} doesn't initialize")

    def set_relation_to_graph(self, extracted_relation: List, document: str) -> None:
        for relation in extracted_relation:
            subj = relation.get("subject", "None subject")
            pred = relation.get("predicate", "None predicate")
            obj = relation.get("object", "None object")
            self.graph.add_edge(subj, obj, label=pred, text=document)

    def set_relation_from_query(self, extracted_relation: List) -> None:
        self.triplets_from_query = []
        try:
            for relation in extracted_relation:
                subj = relation.get("subject", "None subject")
                pred = self.normalize_predicate(relation.get("predicate", "None predicate"))
                obj = relation.get("object", "None object")
                self.triplets_from_query.append([subj, pred, obj])

        except Exception as e:
            logger(f"Cannot set extracted triplet from query [[94]]: {e}")

    def search_relation_from_graph(self,
                      subject_pattern: Optional[str] = None,
                      relation_pattern: Optional[str] = None,
                      object_pattern: Optional[str] = None) -> None:

        self.triplets_from_graph = []

        subject_pattern = r'{s}'.format(s=subject_pattern)
        relation_pattern = r'{r}'.format(r=relation_pattern)
        object_pattern = r'{o}'.format(o=object_pattern)

        try:
            for u, v, data in self.graph.edges(data=True):
                if subject_pattern and not re.search(subject_pattern, str(u), re.IGNORECASE):
                    continue

                pred = data.get('label', '')
                if relation_pattern and not re.search(relation_pattern, str(pred), re.IGNORECASE):
                    continue

                if object_pattern and not re.search(object_pattern, str(v), re.IGNORECASE):
                    continue

                self.triplets_from_graph.append({
                    'subject': u,
                    'predicate': pred,
                    'object': v,
                    'document': data.get('text', "No document")
                })

            if self.config['graph']['limit']:
                self.triplets_from_graph = self.triplets_from_graph[:self.config['graph']['limit']]
            else:
                self.triplets_from_graph = self.triplets_from_graph[:3]
        except Exception as e:
            logger(f"Search triple Error [[90]]: {e}")
        logger(f"Extracted triplets by [SUBJECT, PREDICATE, OBJECT]: {len(self.triplets_from_graph)}")


    def search_relation_by_subject(self, subject: str) -> None:
        self.triplets_from_graph = []
        for u, v, data in self.graph.edges(data=True):
            if subject.lower() in str(u).lower():
                self.triplets_from_graph.append({
                    'subject': u,
                    'predicate': data.get('label', ''),
                    'object': v,
                    'document': data.get('text', "No document")
                })
        if self.config['graph']['limit']:
            self.triplets_from_graph = self.triplets_from_graph[:self.config['graph']['limit']]
        else:
            self.triplets_from_graph = self.triplets_from_graph[:3]
        logger(f"Extracted triplets by [SUBJECT] only: {len(self.triplets_from_graph)}")

    def clean_text(self, text: str) -> str:
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\(\)-]', '', text)
        return text.strip()

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
                if all([triplet['subject'], triplet['predicate'], triplet['object']]):
                    # Clean predicate
                    triplet['predicate'] = self.normalize_predicate(triplet['predicate'])
                    validated.append(triplet)
        except Exception as e:
            logger(f"Validate triplets Error [[92]]: {e}")

        return validated

    def normalize_predicate(self, predicate: str) -> str:
        # Convert to lowercase and replace spaces
        predicate = predicate.lower().strip()
        predicate = re.sub(r'\s+', '_', predicate)
        # Common predicate mappings
        # mappings = {
        #     "is_located_in": "located_in",
        #     "is_a": "is_type_of",
        # }
        # return mappings.get(predicate, predicate)
        return predicate

    def create_inverse_relationships(self, triplets: List[Dict]) -> List[Dict]:
        reverse_triplet = []
        for t in triplets:
            reverse_triplet.append({
                'subject': t['object'],
                'predicate': t['predicate'],
                'object': t['subject']
            })
        return reverse_triplet

    def summary_graph_triplets(self) -> Dict:
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'label_nodes': len([u for u, d in self.graph.nodes(data=True) if d.get('data') == 'label']),
            'document_nodes': len([u for u, d in self.graph.nodes(data=True) if d.get('data') == 'text'])
        }

    def set_documents(self, documents: List[str]) -> None:
        self.documents = documents

    def set_llm_model(self, llm_model: Respondent) -> None:
        self.llm_model = llm_model

    def set_graph(self, graph) -> None:
        self.graph = graph

    def set_config(self, config: Dict) -> None:
        self.config = config

    def get_triplets_from_graph(self) -> List:
        return self.triplets_from_graph

    def get_triplets_from_query(self) -> List:
        return self.triplets_from_query