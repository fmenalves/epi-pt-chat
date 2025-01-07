import json
import os
from langsmith import traceable

from langchain_ollama.llms import OllamaLLM

# https://blog.futuresmart.ai/a-beginners-guide-to-evaluating-rag-systems-with-langsmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

from app.core import present_result_filtered
from langsmith import Client
from langsmith.evaluation import LangChainStringEvaluator, evaluate

###
with open("app/test/dataset.json") as f:
    data = json.load(f)

# Define your QA pairs
inputs = data["Ciplox 500mg"]["questions"]

outputs = data["Ciplox 500mg"]["ground_truths"]

# Create QA pairs
qa_pairs = [{"question": q, "answer": a} for q, a in zip(inputs, outputs)]

# Initialize Langsmith client
client = Client()

# Define dataset parameters
dataset_name = "RAG_Test-1"
dataset_description = "QA pairs about Ciplox 500mg"

try:
    # Create the dataset
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=dataset_description,
    )
    client.create_examples(
        inputs=[{"question": q} for q in inputs],
        outputs=[{"answer": a} for a in outputs],
        dataset_id=dataset.id,
    )
except:
    pass


medication = "Ciplox MG"
strength = "500 mg"

@traceable()
def get_answer(example):
    answer = present_result_filtered(example["question"], medication, strength)
    print("----->" + str(answer))
    return answer


llm = OllamaLLM(
    model="llama3.1:70b",
    base_url="https://gecadllm.fish-albacore.ts.net:8443/api",
    temperature=0,
    request_timeout=150,
)
# Evaluator for comparing RAG answers to reference answers
qa_evaluator = [
    LangChainStringEvaluator(
        "qa",
        config={"llm": llm},
        prepare_data=lambda run, example: {
            "prediction": run.outputs["response"],  # RAG system's answer
            "reference": example.outputs["answer"],  # Ground truth answer
            "input": example.inputs["question"],  # Original question
        },
    )
]

experiment_results = evaluate(
    get_answer,
    data=dataset_name,
    evaluators=qa_evaluator,
    experiment_prefix="rag-qa-oai",
    metadata={"variant": "LCEL context, gpt-4o-mini"},
)


# # Evaluator for detecting hallucinations
# answer_hallucination_evaluator = LangChainStringEvaluator(
#     "labeled_score_string",
#     config={
#         "criteria": {
#             "accuracy": """Is the Assistant's Answer grounded in the Ground Truth documentation?
#             A score of [[1]] means the answer is not at all based on the documentation.
#             A score of [[5]] means the answer contains some information not in the documentation.
#             A score of [[10]] means the answer is fully based on the documentation."""
#         },
#         "normalize_by": 10,  # Normalize scores to a 0-1 scale
#         "llm": model,
#     },
#     prepare_data=lambda run, example: {
#         "prediction": run.outputs["answer"],  # RAG system's answer
#         "reference": run.outputs["contexts"],  # Retrieved documents
#         "input": example.inputs["question"],  # Original question
#     },
# )

# experiment_results = evaluate(
#     predict_rag_answer_with_context,
#     data=dataset_name,
#     evaluators=[answer_hallucination_evaluator],
#     experiment_prefix="rag-qa-oai-hallucination",
#     metadata={"variant": "LCEL context, gpt-4o-mini"},
# )
