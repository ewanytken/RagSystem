from sympy.polys.polyconfig import query

from app.logger.logger_metrics import LoggerMetrics
from app.utils import Utils
from metrics.generation_metrics import GenerationMetrics
from metrics.groundedness.ground_llm import LLMGroundedness

logger_metrics = LoggerMetrics()

class JudgeMetrics(GenerationMetrics):
    def __init__(self):
        super().__init__()

    def set_model_judge(self, model) -> None:
        self.model_judge = model

    def judge_calculation(self):
        completeness = Utils.load_template(self.config['metrics']['prompts']['completeness'])
        correctness = Utils.load_template(self.config['metrics']['prompts']['correctness'])
        faithfulness = Utils.load_template(self.config['metrics']['prompts']['faithfulness'])
        relevance = Utils.load_template(self.config['metrics']['prompts']['relevance'])
        groundedness = LLMGroundedness(self.model_judge, self.config)

        context = " ".join(self.context)

        logger_metrics(f"Answer for Judge LLM: {self.response}\n"
                       f"Query for Judge LLM: {self.query}\n"
                       f"Context for Judge LLM: {context}\n")

        completeness_fill = completeness.format(query=self.query, answer=self.response)
        correctness_fill = correctness.format(context=context, query=self.query, answer=self.response)
        faithfulness_fill = faithfulness.format(context=context, answer=self.response)
        relevance_fill = relevance.format(query=self.query, context=context)

        if self.model_judge:
            self.score['judge_groundedness'] = groundedness.evaluate(response=self.response, context=self.context)
            self.score['judge_completeness'] = self.model_judge.generate(completeness_fill)
            self.score['judge_correctness'] = self.model_judge.generate(correctness_fill)
            self.score['judge_faithfulness'] = self.model_judge.generate(faithfulness_fill) # same groundedness advanced and simple approach for separation
            self.score['judge_relevance'] = self.model_judge.generate(relevance_fill)