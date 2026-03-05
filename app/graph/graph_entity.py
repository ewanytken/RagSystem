from typing import List, Dict, Optional, Union
import networkx as nx
from app.logger import LoggerWrapper
import hashlib

logger = LoggerWrapper()

"""
Input: entities - List[Dict]. Use simple nx.Graph  
Output: list of dict: entity, label, score
"""

class GraphEntity:

    def __init__(self):
        self.graph = nx.Graph()
        self.entities: Optional[List[Dict]] = []

    def add_to_knowledge_graph(self, document: str) -> None:
        try:
            doc_node = GraphEntity.hash_maker(document)
            self.graph.add_node(doc_node, type='document', text=document)

            for entity in self.entities:
                entity_node = GraphEntity.hash_maker(entity['entity'])

                self.graph.add_node(
                    entity_node,
                    type='entity',
                    entity=entity['entity'], # field "text" from gliner extraction
                    label=entity['label'],
                )
                self.graph.add_edge(
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
                            })
            except Exception as e:
                logger(f"Find related entities error [[101]]: {e}")

        return related

    def get_knowledge_graph_stats(self) -> Dict:
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'entity_nodes': len([n for n, d in self.graph.nodes(data=True) if d.get('type') == 'entity']),
            'document_nodes': len([n for n, d in self.graph.nodes(data=True) if d.get('type') == 'document'])
        }

    def advanced_word_search(self, search_entity, attribute_name='entity') -> Dict[str, Dict]:
        try:
            nodes_with_entity = [n for n, attr in self.graph.nodes(data=True)
                               if attr.get(attribute_name) == search_entity]

            if not nodes_with_entity:
                return {}

            results = {}
            for node in nodes_with_entity:
                neighbors = list(self.graph.neighbors(node))

                neighbor_details = []
                for neighbor in neighbors:
                    edge_data = self.graph.get_edge_data(node, neighbor)
                    neighbor_details.append({
                        'node': neighbor,
                        'entity': self.graph.nodes[neighbor].get(attribute_name, 'Unknown'),
                        'edge_attributes': edge_data,
                        'neighbor_degree': self.graph.degree(neighbor)
                    })

                results[node] = {
                    'entity': search_entity,
                    'node_id': node,
                    'neighbors': neighbor_details,
                    'degree': self.graph.degree(node),
                    'clustering': nx.clustering(self.graph, node),
                    'node_attributes': self.graph.nodes[node]
                }
            # doc_node = GraphEntity.hash_maker(document)
            # self.graph.add_node(doc_node, type='document', text=document)
            #
            # for entity in self.entities:
            #     entity_node = GraphEntity.hash_maker(entity['entity'])
            #
            #     self.graph.add_node(
            #         entity_node,
            #         type='entity',
            #         entity=entity['entity'], # field "text" from gliner extraction
            #         label=entity['label'],
            #     )
            #     self.graph.add_edge(
            #         doc_node, entity_node,
            #         relation='contains',
            #         weight = 1
            #     )
            return results
        except Exception as e:
            logger(f"Advanced word search error [[102]]: {e}")

    def set_entities(self, entities) -> None:
        self.entities = entities

    @staticmethod
    def hash_maker(text: str) -> str:
        hash_object = hashlib.sha256(text.encode('utf-8'))
        return hash_object.hexdigest()