from typing import List, Dict, Optional
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

    def set_entities(self, entities) -> None:
        self.entities = entities

    def add_to_knowledge_graph(self, document: str) -> None:

        doc_node = f"doc_{hash(document) % 1000000}"
        self.knowledge_graph.add_node(doc_node, type='document', text=document[:100])

        for entity in self.entities:
            entity_node = f"ent_{hash(entity['text']) % 1000000}"

            self.knowledge_graph.add_node(
                entity_node,
                type='entity',
                text=entity['text'],
                label=entity['label'],
                score=entity['score']
            )
            self.knowledge_graph.add_edge(
                doc_node, entity_node,
                relation='contains',
                weight=entity['score']
            )

    def find_related_entities(self, entity_text: str, limit: int = 15) -> List[Dict]:

        related = []
        entity_hash = f"ent_{hash(entity_text) % 1000000}"

        if entity_hash in self.knowledge_graph:
            neighbors = list(self.knowledge_graph.neighbors(entity_hash))

            for neighbor in neighbors[:limit]:
                node_data = self.knowledge_graph.nodes[neighbor]

                if node_data.get('type') == 'entity':
                    related.append({
                        'entity': node_data['text'],
                        'label': node_data['label'],
                        'score': node_data.get('score', 0.5)
                    })
        return related

    def get_knowledge_graph_stats(self) -> Dict:
        return {
            'nodes': self.knowledge_graph.number_of_nodes(),
            'edges': self.knowledge_graph.number_of_edges(),
            'entity_nodes': len([n for n, d in self.knowledge_graph.nodes(data=True) if d.get('type') == 'entity']),
            'document_nodes': len([n for n, d in self.knowledge_graph.nodes(data=True) if d.get('type') == 'document'])
        }