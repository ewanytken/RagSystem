from app.logger.logger_metrics import LoggerMetrics
from app.utils import Utils
from metrics.generation_metrics import GenerationMetrics

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

        answer = " ".join(self.answers)
        query = " ".join(self.queries)
        context = " ".join(self.contexts)

        logger_metrics(f"Answer for Judge LLM: {answer}\n"
                       f"Query for Judge LLM: {query}\n"
                       f"Context for Judge LLM: {context}\n")

        completeness_fill = completeness.format(query=query, answer=answer)
        correctness_fill = correctness.format(context=context, query=query, answer=answer)
        faithfulness_fill = faithfulness.format(context=context, answer=answer)
        relevance_fill = relevance.format(query=query, context=context)

        if self.model_judge:
            self.score['judge_completeness'] = self.model_judge.generate(completeness_fill)
            self.score['judge_correctness'] = self.model_judge.generate(correctness_fill)
            self.score['judge_faithfulness'] = self.model_judge.generate(faithfulness_fill)
            self.score['judge_relevance'] = self.model_judge.generate(relevance_fill)