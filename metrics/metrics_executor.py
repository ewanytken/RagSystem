from typing import Optional, Dict

from app.logger.logger_metrics import LoggerMetrics
from metrics.generation_metrics import GenerationMetrics
from metrics.judge_metrics import JudgeMetrics
from metrics.retrieved_metrics import RetrievedMetrics

logger_metrics = LoggerMetrics()

class MetricsExecutor:
    def __init__(self, config_eval: Dict):
        self.config_eval: Optional[Dict] = config_eval

        self.generate_metric: Optional[GenerationMetrics] = GenerationMetrics()
        self.retrieve_metric: Optional[RetrievedMetrics] = RetrievedMetrics()
        self.judge_metric: Optional[JudgeMetrics] = JudgeMetrics()

        self.generate_metric.init_processing()
        self.overall_scores: Optional[Dict] = {}

    def generation_evaluator(self) -> None:

        if self.generate_metric.get_candidate():
            try:
                self.generate_metric.set_answers(self.config_eval['response'])
                self.generate_metric.set_queries(self.config_eval['query'])
                self.generate_metric.set_contexts(self.config_eval['context'])

                self.generate_metric.generation_calculation()
                self.generate_metric.bert_calculation()
                self.generate_metric.show_scores()

                self.overall_scores.update({"GENERATION_EVAL": self.generate_metric.get_score()})
            except Exception as e:
                logger_metrics(f"Generation metrics ERROR {e}")
        else:
            logger_metrics(f"Generation metrics pass, don't specify candidates")

    def retriever_evaluator(self) -> None:

        if self.retrieve_metric.get_relevant_doc():
            try:
                self.retrieve_metric.set_retrieved_docs(self.config_eval['retrieved_docs'])
                self.retrieve_metric.retriever_calculation()
                self.retrieve_metric.show_scores()
                self.overall_scores.update({"RETRIEVED_EVAL": self.retrieve_metric.get_score()})
            except Exception as e:
                logger_metrics(f"Retrieval metrics ERROR {e}")
        else:
            logger_metrics(f"Retrieval metrics pass, don't specify relevant docs")

    def judge_evaluator(self) -> None:
        try:
            self.judge_metric.set_model_judge(self.config_eval['judge_model'])
            self.judge_metric.judge_calculation()
            self.judge_metric.show_scores()
            self.overall_scores.update({"JUDGE_EVAL": self.judge_metric.get_score()})
        except Exception as e:
            logger_metrics(f"Judge LLM metrics ERROR {e}")

    def get_overall_scores(self) -> Dict:
        return self.overall_scores

    def set_candidate(self, candidates: str) -> None:
        self.generate_metric.set_candidates(candidates)

    def set_relevant_context(self, relevant_docs: str) -> None:
        self.generate_metric.set_relevant_docs(relevant_docs)