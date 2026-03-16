from typing import List, Dict
import numpy as np
from app.logger.logger_metrics import LoggerMetrics
from metrics.metrics import Metrics

logger = LoggerMetrics()

class RetrievedMetrics(Metrics):
    def __init__(self):
        super().__init__()
        self.retrieved_docs = []
        self.relevant_docs = []
        self.k = 5

    def retriever_calculation(self) -> None:
        retrieved_at_k = self.retrieved_docs[:self.k]
        relevant_retrieved = [doc for doc in retrieved_at_k if doc in self.relevant_docs]
        self.score["Precision@K"] = len(relevant_retrieved) / self.k
        self.score["Recall@K"] = len(relevant_retrieved) / len(self.relevant_docs)

        reciprocal_ranks = []
        for retrieved, relevant in zip(self.retrieved_docs, self.relevant_docs):
            for i, doc in enumerate(retrieved, 1):
                if doc in relevant:
                    reciprocal_ranks.append(1/i)
                    break
            else:
                reciprocal_ranks.append(0)
        self.score["MRR"] = sum(reciprocal_ranks) / len(reciprocal_ranks)

    def ndcg_at_k(self, relevance_scores: List) -> None:
        dcg = sum((2 ** rel - 1) / np.log2(i + 2)
                  for i, rel in enumerate(relevance_scores[:self.k]))

        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = sum((2 ** rel - 1) / np.log2(i + 2)
                   for i, rel in enumerate(ideal_scores[:self.k]))

        self.score["NDCG"] =  dcg / idcg if idcg > 0 else 0

    def set_relevant_docs(self, docs: List) -> None:
        self.retrieved_docs = docs

    def set_retrieved_docs(self, docs: List) -> None:
        self.relevant_docs = docs

    def set_k(self, k: int) -> None:
        self.k = k

    def get_score(self) -> Dict:
        return self.score