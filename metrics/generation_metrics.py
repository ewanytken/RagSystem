from typing import List

import numpy as np
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu

from app.logger import LoggerWrapper
from app.logger.logger_metrics import LoggerMetrics
from metrics.groundedness.ground_base import RuleBasedGroundedness
from metrics.metrics import Metrics

logger_metrics = LoggerMetrics()
logger = LoggerWrapper()

class GenerationMetrics(Metrics):

    def __init__(self):
        super().__init__()
        self.query: str = ""
        self.response: str = ""
        self.context: List[str] = []

    def generation_calculation(self) -> None:
        logger(f"Setup Query (GenerationMetrics - generation_calc): {True if self.response else False}")
        logger(f"Setup Response (GenerationMetrics - generation_calc): {True if self.candidate else False}")

        if self.candidate:
            try:
                self.score["BLEU"] = sentence_bleu([self.response.split()], self.candidate.split())
                logger_metrics(f"BLEU {self.score["BLEU"]}")

                # self.score["METEOR"] =  meteor_score([self.answers], self.candidates)
                P, R, F1 = score([self.candidate], [self.response], lang=self.config['metrics']['bert_lang'])
                self.score["BERT_SCORE"] =  F1.mean().item()
                logger_metrics(f"BERT_SCORE {self.score["BERT_SCORE"]}")

            except Exception as e:
                logger_metrics(f"BLEU or BERT_SCORE cause ERROR [[140]] {e}")
        else:
            logger_metrics(f"Candidates doesn't initialize {len(self.candidate)}")

    def bert_calculation(self) -> None:

        logger(f"Setup Query (GenerationMetrics - bert_calc): {True if self.query else False}")
        logger(f"Setup Response (GenerationMetrics - bert_calc): {True if self.response else False}")
        logger(f"Setup Context {type(self.context)} (GenerationMetrics - bert_calc): {True if self.context else False}")

        try:
            scores = []
            for query, answer in zip(self.query, self.response):
                q_emb = self.model_sim.encode(query)
                a_emb = self.model_sim.encode(answer)

                q_emb = q_emb / np.linalg.norm(q_emb)
                a_emb = a_emb / np.linalg.norm(a_emb)

                similarity = np.dot(q_emb, a_emb)
                scores.append(similarity)

            self.score["Answer_relevance"] = np.mean(scores)
            logger_metrics(f"Answer relevance {self.score["Answer_relevance"]}")

            scores = []
            for query, context_list in zip(self.query, self.context):
                q_emb = self.model_sim.encode(query)

                context_scores = []
                for context in context_list:
                    c_emb = self.model_sim.encode(context)
                    c_emb = c_emb / np.linalg.norm(c_emb)
                    context_scores.append(np.dot(q_emb, c_emb))
                scores.append(np.mean(context_scores))

            self.score["Context_relevance"] = np.mean(scores)
            logger_metrics(f"Context relevance {self.score["Context_relevance"]}")

            rule_groundedness = RuleBasedGroundedness(threshold=0.3)
            result = rule_groundedness.evaluate(self.response, self.context)

            self.score["Groundedness_score"] = result.score
            logger_metrics(f"Groundedness score: {self.score["Groundedness_score"]}")
            logger_metrics(f"Groundedness details: {result.details}")

        except Exception as e:
            logger(f"Answer, context relevance or groundedness metrics caused ERROR: {e}")

    def set_queries(self, queries) -> None:
        self.query = queries

    def set_answers(self, response) -> None:
        self.response = response

    def set_contexts(self, contexts) -> None:
        self.context = contexts
