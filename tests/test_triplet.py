import unittest
from typing import Dict

import networkx as nx

from app.documents_processor.word_handler import WordPdfHandler
from app.graph.triplet_extractor import TripletExtractor
from app.logger import LoggerWrapper
from app.respondent.external_model.respondent_other_service import ExternalModel
from app.utils import Utils

logger = LoggerWrapper()

class TestRAGSystem(unittest.TestCase):



    # def semantic_search_by_query(self, query_text: str, top_k: int = 5) -> List[Dict]:
    #     self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    #
    #     if self.triples:
    #         self.triple_embeddings = self.encoder.encode([
    #             t['text'] for t in self.triples
    #         ])
    #     query_embedding = self.encoder.encode([query_text])
    #
    #     if not self.triple_embeddings.size:
    #         return []
    #
    #     similarities = cosine_similarity(query_embedding, self.triple_embeddings)[0]
    #     top_indices = np.argsort(similarities)[-top_k:][::-1]
    #
    #     results = []
    #     for idx in top_indices:
    #         triple = self.triples[idx].copy()
    #         triple['similarity'] = float(similarities[idx])
    #         results.append(triple)
    #
    #     return results
    #
    # def visualize_word_neighbors(self, search_word):
    #
    #     import matplotlib.pyplot as plt
    #
    #     # Find node(s) with the word
    #     nodes_with_word = [n for n, attr in self.graph.nodes(data=True)
    #                        if attr.get('word') == search_word]
    #
    #     if not nodes_with_word:
    #         print(f"Word '{search_word}' not found")
    #         return
    #
    #     for node in nodes_with_word:
    #         # Get ego network (node + its neighbors)
    #         neighbors = list(self.graph.neighbors(node))
    #         subgraph_nodes = [node] + neighbors
    #
    #         # Create subgraph
    #         subgraph = self.graph.subgraph(subgraph_nodes)
    #
    #         # Visualize
    #         pos = nx.spring_layout(subgraph)
    #         nx.draw(subgraph, pos, with_labels=True, node_color='lightblue',
    #                 node_size=500, font_size=10, font_weight='bold')
    #
    #         # Highlight the search word node
    #         nx.draw_networkx_nodes(subgraph, pos, nodelist=[node],
    #                                node_color='red', node_size=600)
    #
    #         plt.title(f"Neighbors of '{search_word}'")
    #         plt.show()
    def summary_graph_triplets(self, G) -> Dict:
        return {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'label_nodes': len([u for u, d in G.nodes(data=True) if d.get('data') == 'label']),
            'document_nodes': len([u for u, d in G.nodes(data=True) if d.get('data') == 'text'])
        }

    def visualize_word_neighbors(self, G,  search_word):

        import matplotlib.pyplot as plt

        # Find node(s) with the word
        nodes_with_word = [n for n, d in G.nodes(data=True)
                           if search_word in n]

        if not nodes_with_word:
            print(f"Word '{search_word}' not found")
            return

        for node in nodes_with_word:
            # Get ego network (node + its neighbors)
            neighbors = list(G.neighbors(node))
            subgraph_nodes = [node] + neighbors

            # Create subgraph
            subgraph = G.subgraph(subgraph_nodes)

            # Visualize
            pos = nx.spring_layout(subgraph)
            nx.draw(subgraph, pos, with_labels=True, node_color='lightblue',
                    node_size=500, font_size=10, font_weight='bold')

            # Highlight the search word node
            nx.draw_networkx_nodes(subgraph, pos, nodelist=[node],
                                   node_color='red', node_size=100)

            plt.title(f"Neighbors of '{search_word}'")
            plt.show()

    def visualize_graph(self, graph):

        import matplotlib.pyplot as plt

        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_color='lightblue',
                node_size=30, font_size=5)

        nx.draw_networkx_nodes(graph, pos, nodelist=list(self.triplet.graph),
                               node_color='red', node_size=30)

        plt.title(f"Neighbors of '{graph}'")
        plt.show()

    def setUp(self):
        self.triplet = TripletExtractor()

        self.word_handler = WordPdfHandler()

    def test_document_processing(self):
        config = Utils.get_config_file()

        self.word_handler.set_config(config)
        self.word_handler.handle_documents()
        doc_in_chunks = self.word_handler.get_chunked_documents()
        print(len(doc_in_chunks))
        self.triplet.set_config(config)
        self.triplet.set_documents(doc_in_chunks[:1])
        # model = TransformerWrapper()
        model = ExternalModel()

        self.triplet.set_llm_model(model)
        self.triplet.extract_triplets()

        print(self.summary_graph_triplets(self.triplet.graph))


        self.visualize_graph(self.triplet.graph)

        # self.visualize_word_neighbors(self.triplet.graph, "Общество")

        # query = "Кто является генеральным директором АО «Селектел»?"
        # self.triplet.extract_triplets(query)
        # triplet_query = self.triplet.get_triplets_from_query()
        # logger(self.triplet.get_triplets_from_query())
        # for triplet in triplet_query:
        #     self.triplet.search_relation_by_subject(triplet[0])
        #     logger(self.triplet.get_triplets_from_graph())
        #
        #     self.triplet.search_relation_from_graph(triplet[0], triplet[1], triplet[2])
        #     logger(self.triplet.get_triplets_from_graph())

if __name__ == '__main__':
    unittest.main()