from typing import List, Dict, Optional, Union, Set
import networkx as nx
from app.logger import LoggerWrapper
import hashlib

logger = LoggerWrapper()

"""
Input: entities - List[Dict]. Use simple nx.Graph  
Output: list of dict for entity: entity, label, score. Set of text for document search
"""

class GraphEntity:

    def __init__(self) -> None:
        self.graph = nx.Graph()
        self.entities: Optional[List[Dict]] = []

    def __repr__(self):
        return f"Graph Entity Component"

    def add_to_knowledge_graph(self, document: str) -> None:
        try:
            doc_node = GraphEntity.hash_maker(document)

            self.graph.add_node(doc_node,
                                type='document',
                                text=document)

            for entity in self.entities:
                entity_node = GraphEntity.hash_maker(entity['entity'])

                self.graph.add_node(
                    entity_node,
                    type='entity',
                    entity=entity['entity'],
                    label=entity['label'],
                    score=entity['score']
                )
                self.graph.add_edge(
                    doc_node, entity_node,
                    relation='contains',
                    weight = entity['score'],
                )

        except Exception as e:
            logger(f"Failed to add entity to knowledge graph [[100]]: {e}")

    def find_related_entities_from_doc(self, documents: Union[str, List], limit: int = 150) -> List[Dict]:
        related = []
        if isinstance(documents, str): documents = [documents]
        for doc in documents:
            doc_node = GraphEntity.hash_maker(doc)
            try:
                if doc_node in self.graph:
                    neighbors = list(self.graph.neighbors(doc_node))

                    for neighbor in neighbors[:limit]:
                        node_data = self.graph.nodes[neighbor]

                        if node_data.get('type') == 'entity':
                            related.append({
                                'entity': node_data['entity'],
                                'label': node_data['label'],
                                'score': node_data['score']
                            })

            except Exception as e:
                logger(f"Find related entities error [[101]]: {e}")
        logger(f"Extracted entities from Graph: {len(related)}")
        return related

    def find_doc_by_entity(self, search_entity: str, attribute_name='entity') -> Set[Dict[str, Union[str, int]]]:
        try:
            nodes_with_entity = [n for n, attr in self.graph.nodes(data=True)
                                            if attr.get(attribute_name) == search_entity]
            results = set()

            if not nodes_with_entity:
                return results

            for node in nodes_with_entity: # node is a hash
                neighbors = list(self.graph.neighbors(node))
                for neighbor in neighbors:
                    edge_data = self.graph.get_edge_data(node, neighbor)
                    if neighbor.get('type') == 'document':
                        results.update({
                            'document': neighbor['text'],
                            'score': edge_data['weight']
                        })
            return results
        except Exception as e:
            logger(f"Don't find document by entity ERROR [[102]]: {e}")

    def set_entities(self, entities) -> None:
        self.entities = entities

    def get_knowledge_graph_stats(self) -> Dict:
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'entity_nodes': len([n for n, d in self.graph.nodes(data=True) if d.get('type') == 'entity']),
            'document_nodes': len([n for n, d in self.graph.nodes(data=True) if d.get('type') == 'document'])
        }

    @staticmethod
    def hash_maker(text: str) -> str:
        hash_object = hashlib.sha256(text.encode('utf-8'))
        return hash_object.hexdigest()