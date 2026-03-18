from typing import Optional, List

import numpy as np
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score

from app.logger.logger_metrics import LoggerMetrics
from metrics.metrics import Metrics

logger_metrics = LoggerMetrics()

class GenerationMetrics(Metrics):
    def __init__(self):
        super().__init__()
        self.queries: Optional[List[str]] = []
        self.answers: Optional[List[str]] = []
        self.contexts: Optional[List[str]] = []

    def generation_calculation(self) -> None:
        logger_metrics(f"Setup Queries: {True if self.answers else False}")
        logger_metrics(f"Setup Answer: {True if self.candidates else False}")
        if self.candidates:
            try:
                self.score["BLEU"] = sentence_bleu(self.answers, self.candidates)
                self.score["METEOR"] =  meteor_score(self.answers, self.candidates)

                P, R, F1 = score(self.candidates, self.answers, lang=self.lang)
                self.score["BEST_SCORE"] =  F1.mean().item()
            except Exception as e:
                logger_metrics(f"Generation metrics ERROR [[]]{e}")
        else:
            logger_metrics(f"Candidates doesn't initialize {len(self.candidates)}")

    def bert_calculation(self) -> None:

        logger_metrics(f"Setup Queries: {True if self.queries else False}")
        logger_metrics(f"Setup Answer: {True if self.answers else False}")
        logger_metrics(f"Setup Context: {True if self.contexts else False}")

        try:
            scores = []
            for query, answer in zip(self.queries, self.answers):
                q_emb = self.model_sim.encode(query)
                a_emb = self.model_sim.encode(answer)

                q_emb = q_emb / np.linalg.norm(q_emb)
                a_emb = a_emb / np.linalg.norm(a_emb)

                similarity = np.dot(q_emb, a_emb)
                scores.append(similarity)

            self.score["Answer_Relevance"] = np.mean(scores)

            scores = []
            for query, context_list in zip(self.queries, self.contexts):
                q_emb = self.model_sim.encode(query)

                context_scores = []
                for context in context_list:
                    c_emb = self.model_sim.encode(context)
                    c_emb = c_emb / np.linalg.norm(c_emb)
                    context_scores.append(np.dot(q_emb, c_emb))

                scores.append(np.mean(context_scores))

            self.score["Context_Relevance"] = np.mean(scores)

            def extract_claims(text):
                # Simplified claim extraction
                sentences = text.split('.')
                return [s.strip() for s in sentences if len(s.strip()) > 10]

            def verify_claim(claim, cont):
                # Simplified verification
                context_text = ' '.join(cont)
                # Check if key terms appear in context
                claim_words = set(claim.lower().split())
                context_words = set(context_text.lower().split())
                overlap = len(claim_words & context_words) / len(claim_words)
                return overlap > 0.3  # threshold

            scores = []
            for answer, context in zip(self.answers, self.contexts):
                claims = extract_claims(answer)
                if not claims:
                    continue

                supported = sum(verify_claim(claim, context) for claim in claims)
                scores.append(supported / len(claims))

            self.score["Groundedness"] =  np.mean(scores)

        except Exception as e:
            logger_metrics(f"Relevance and RAGAS metrics ERROR {e}")

    def set_queries(self, queries) -> None:
        self.queries = queries

    def set_answers(self, answers) -> None:
        self.answers = answers

    def set_contexts(self, contexts) -> None:
        self.contexts = contexts

    def set_lang(self, lang) -> None:
        self.lang = lang
