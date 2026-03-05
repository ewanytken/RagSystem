from typing import List, Dict, Optional, Union
import networkx as nx

from app.logger import LoggerWrapper

logger = LoggerWrapper()

"""
Input: entities - List[Dict]. Use simple nx.Graph  
Output: list of dict: entity, label, score
"""

class GraphEntity:

    def __init__(self):
        self.knowledge_graph = nx.Graph()
        self.entities: Optional[List[Dict]] = []

    def add_to_knowledge_graph(self, document: str) -> None:
        try:
            doc_node = f"doc_{hash(document) % 1000000}"
            self.knowledge_graph.add_node(doc_node, type='document', text=document)

            for entity in self.entities:
                entity_node = f"ent_{hash(entity['entity']) % 1000000}"

                self.knowledge_graph.add_node(
                    entity_node,
                    type='entity',
                    text=entity['entity'], # field "text" from gliner extraction
                    label=entity['label'],
                )
                self.knowledge_graph.add_edge(
                    doc_node, entity_node,
                    relation='contains',
                    weight = 1
                )
        except Exception as e:
            logger(f"Failed to add entity to knowledge graph [[100]]: {e}")

    def find_related_entities(self, documents: Union[str, List], limit: int = 150) -> List[Dict]:
        related = []
        if isinstance(documents, str): documents = [documents]
        for doc in documents:
            doc_node = f"ent_{hash(doc) % 1000000}"
            try:
                if doc_node in self.knowledge_graph:
                    neighbors = list(self.knowledge_graph.neighbors(doc_node))

                    for neighbor in neighbors[:limit]:
                        node_data = self.knowledge_graph.nodes[neighbor]

                        if node_data.get('type') == 'entity':
                            related.append({
                                'entity': node_data['entity'],
                                'label': node_data['label'],
                            })
            except Exception as e:
                logger(f"Find related entities error [[101]]: {e}")

        return related

    def get_knowledge_graph_stats(self) -> Dict:
        return {
            'nodes': self.knowledge_graph.number_of_nodes(),
            'edges': self.knowledge_graph.number_of_edges(),
            'entity_nodes': len([n for n, d in self.knowledge_graph.nodes(data=True) if d.get('type') == 'entity']),
            'document_nodes': len([n for n, d in self.knowledge_graph.nodes(data=True) if d.get('type') == 'document'])
        }

    def set_entities(self, entities) -> None:
        self.entities = entities