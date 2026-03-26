from typing import List

import numpy as np

from app.logger import LoggerWrapper
from app.logger.logger_metrics import LoggerMetrics
from metrics.metrics import Metrics

logger_metrics = LoggerMetrics()
logger = LoggerWrapper()

class RetrievedMetrics(Metrics):
    def __init__(self):
        super().__init__()
        self.retrieved_docs = []
        self.k = 10

    def retriever_calculation(self) -> None:
        if self.retrieved_docs and self.relevant_doc:
            try:
                retrieved_at_k = self.retrieved_docs[:self.k]
                relevant_retrieved = [doc for doc in retrieved_at_k if doc in self.relevant_doc]
                self.score["Precision@K"] = len(relevant_retrieved) / self.k if len(relevant_retrieved) else "Cannot calculate Precision@K"
                logger_metrics(f"Precision@K {self.score["Precision@K"]}. Relevant_retrieved: {len(relevant_retrieved)}")

                self.score["Recall@K"] = len(relevant_retrieved) / len(self.relevant_doc) if len(relevant_retrieved) else "Cannot calculate Recall@K"
                logger_metrics(f"Recall@K {self.score["Recall@K"]}. Relevant_retrieved: {len(relevant_retrieved)}")

                reciprocal_ranks = []
                for retrieved, relevant in zip(self.retrieved_docs, self.relevant_doc):
                    for i, doc in enumerate(retrieved, 1):
                        if doc in relevant:
                            reciprocal_ranks.append(1/i)
                            break
                    else:
                        reciprocal_ranks.append(0)
                self.score["MRR"] = sum(reciprocal_ranks) / len(reciprocal_ranks)
                logger_metrics(f"MRR {self.score["MRR"]}. Reciprocal_ranks: {len(reciprocal_ranks)}")

            except Exception as e:
                logger(f"Retrieved metrics caused an ERROR {e}")

    def ndcg_at_k(self, relevance_scores: List) -> None:
        dcg = sum((2 ** rel - 1) / np.log2(i + 2)
                  for i, rel in enumerate(relevance_scores[:self.k]))

        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = sum((2 ** rel - 1) / np.log2(i + 2)
                   for i, rel in enumerate(ideal_scores[:self.k]))

        self.score["NDCG"] =  dcg / idcg if idcg > 0 else 0

    def set_k(self, k: int) -> None:
        self.k = k

    def set_retrieved_docs(self, docs: List) -> None:
        self.retrieved_docs = docs
