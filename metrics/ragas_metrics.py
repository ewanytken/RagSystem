# from metrics.metrics import Metrics
# import asyncio
# from ragas.metrics.collections import AspectCritic
# from ragas.llms import llm_factory
#
# class MetricsWithRagas(Metrics):
#     def __init__(self):
#         super().__init__()
#
#
# # Setup your LLM
# llm = llm_factory("gpt-4o")
#
# # Create a metric
# metric = AspectCritic(
#     name="summary_accuracy",
#     definition="Verify if the summary is accurate and captures key information.",
#     llm=llm
# )
#
# # Evaluate
# test_data = {
#     "user_input": "summarise given text\nThe company reported an 8% rise in Q3 2024, driven by strong performance in the Asian market. Sales in this region have significantly contributed to the overall growth. Analysts attribute this success to strategic marketing and product localization. The positive trend in the Asian market is expected to continue into the next quarter.",
#     "response": "The company experienced an 8% increase in Q3 2024, largely due to effective marketing strategies and product adaptation, with expectations of continued growth in the coming quarter.",
# }
#
# score = await metric.ascore(
#     user_input=test_data["user_input"],
#     response=test_data["response"]
# )
# print(f"Score: {score.value}")
# print(f"Reason: {score.reason}")