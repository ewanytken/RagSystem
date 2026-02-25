from typing import Optional, Dict, List

import torch
from txtai import Embeddings

from app.logger import LoggerWrapper

logger = LoggerWrapper

"""
Input: config.yaml with path to ticket embedding model [models][embedding], [rag][retrieval_limit] 
Output: List[Dict] with retrieved documents 
"""

class EmbeddingObject:

    def __init__(self):
        self.embeddings: Optional[Embeddings] = None
        self.config: Optional[Dict] = None
        self.retrieve_documents: Optional[List[Dict]] = []

    def set_embedding_model(self) -> None:
        try:
            model_ticker = self.config['models']['embedding']
            self.embeddings = Embeddings({
                "path": model_ticker,
                "gpu": torch.cuda.is_available(),
                "content": True,
                "batch_size": 32
            })
            logger(f"Model loaded {model_ticker} on {'GPU' if torch.cuda.is_available() else 'CPU'}. Batch size by default is 32")
        except Exception as e:
            logger(f"Embedding model install ERROR 60: {e}")
            raise

    def documents_indexing(self, handled_documents: List[str]):
        try:
            self.embeddings.index([(i, text, None) for i, text in enumerate(handled_documents)])
            logger(f"Indexing text block from documents (List[str]): {len(handled_documents)}")

        except Exception as e:
            logger(f"Documents Indexing Error 61: {e}")

    def documents_retrieve(self, user_query: str):
        limit = self.config['rag']['retrieval_limit']
        try:
            results = self.embeddings.search(user_query, limit=limit * 2)

            seen_texts = set()

            for res in results:
                text_hash = hash(res['text'][:1000])

                if text_hash not in seen_texts:
                    seen_texts.add(text_hash)

                    self.retrieve_documents.append({
                        'text': res['text'],
                        'score': res['score'],
                        'id': res['id']
                    })
                if len(self.retrieve_documents) >= limit:
                    break

            logger(f"Documents retrieved {len(self.retrieve_documents)}")

        except Exception as e:
            logger(f"Documents retrieve Error 62: {e}")

    def set_config(self, config: Dict):
        self.config = config

    def get_retrieval_documents(self):
        return self.retrieve_documents