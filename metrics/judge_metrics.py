from app.logger import LoggerWrapper
from app.logger.logger_metrics import LoggerMetrics
from app.utils import Utils
from metrics.generation_metrics import GenerationMetrics
from metrics.groundedness.ground_llm import LLMGroundedness

logger_metrics = LoggerMetrics()
logger = LoggerWrapper()

class JudgeMetrics(GenerationMetrics):
    def __init__(self):
        super().__init__()

    def set_model_judge(self, model) -> None:
        self.model_judge = model

    def judge_calculation(self):
        logger(f"Answer for Judge LLM: {True if self.response else False}\n")
        logger(f"Query for Judge LLM: {True if self.query else False}\n")
        logger(f"Context for Judge LLM: {True if self.context else False}\n")
        logger(f"Number of retrieved documents: {len(self.context)}\n")

        if self.query and self.response and self.context:

            completeness = Utils.load_template(self.config['metrics']['prompts']['completeness'])
            correctness = Utils.load_template(self.config['metrics']['prompts']['correctness'])
            faithfulness = Utils.load_template(self.config['metrics']['prompts']['faithfulness'])
            relevance = Utils.load_template(self.config['metrics']['prompts']['relevance'])
            groundedness = LLMGroundedness(self.model_judge, self.config)

            context_str_all = "PASSAGE:\n".join(self.context)

            completeness_fill = completeness.format(query=self.query, answer=self.response)
            correctness_fill = correctness.format(context=context_str_all, query=self.query, answer=self.response)
            faithfulness_fill = faithfulness.format(context=context_str_all, answer=self.response)
            relevance_fill = relevance.format(query=self.query, context=context_str_all)

            if self.model_judge:
                self.score['judge_groundedness'] = groundedness.evaluate(response=self.response, context=self.context)
                logger_metrics(f"Judge Groundedness {self.score["judge_groundedness"]}")

                self.score['judge_completeness'] = self.model_judge.generate(completeness_fill)
                logger_metrics(f"Judge Completeness {self.score["judge_completeness"]}")

                self.score['judge_correctness'] = self.model_judge.generate(correctness_fill)
                logger_metrics(f"Judge Correctness {self.score["judge_correctness"]}")

                self.score['judge_faithfulness'] = self.model_judge.generate(faithfulness_fill) # same as groundedness, advanced and simple approach for separation
                logger_metrics(f"Judge Faithfulness {self.score["judge_faithfulness"]}")

                self.score['judge_relevance'] = self.model_judge.generate(relevance_fill)
                logger_metrics(f"Judge Relevance {self.score["judge_relevance"]}")
            else:
                logger(f"Judge model don't install")