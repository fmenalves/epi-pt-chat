import json
import os

from langchain_ollama.llms import OllamaLLM

# https://blog.futuresmart.ai/a-beginners-guide-to-evaluating-rag-systems-with-langsmith
# https://docs.ragas.io/en/stable/getstarted/evals/#evaluation
# https://www.giskard.ai/products/llm-evaluation-hub
# https://docs.smith.langchain.com/
# https://medium.com/syntonize/intro-to-ai-evaluation-with-langsmith-f85799788f51
# https://docs.confident-ai.com/docs/guides-rag-evaluation#e2e-rag-evaluation
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

from app.core import present_result_filtered
from langsmith import Client
from langsmith.evaluation import LangChainStringEvaluator, evaluate

###
with open("app/test/dataset.json") as f:
    data = json.load(f)

# Define dataset parameters
dataset_name = "RAG_Test-4"
combined_dataset_description = "A comprehensive QA dataset combining multiple drugs' question-answer pairs with metadata."

DATA_TRANSFORM = {
    "Ciplox 500mg": ["Ciplox MG", "500 mg"],
    "RInvoq 15mg": ["Rinvoq", "15 mg"],
    "Ozempic 0.5mg": ["Ozempic", "0.5 mg/0.37 ml"],
    "Paclitaxel Accord 6mg": ["Paclitaxel Accord MG", "6 mg/ml"],
    "Diovan 80 mg": ["Diovan", "80 mg"],
    "Influvac tetra": ["Influvac Tetra", "Associação"],
    "Comirnaty JN.1": ["Comirnaty JN.1", "30 µg/0.3 ml"],
    "Triticum 100 mg": ["Triticum", "100 mg"],
    "Depakine 40 mg /ml": ["Depakine", "40 mg/ml"],
    "Lenalidomida tecnigen 10mg": ["Lenalidomida Tecnigen MG", "10 mg"],
}


# Initialize Langsmith client
client = Client()
# Prepare all examples with metadata
examples = []
for main_key, content in data.items():
    inputs = content["questions"]
    outputs = content["ground_truths"]

    # Add each question-answer pair with metadata
    examples.extend(
        [
            {
                "inputs": {"question": question},
                "outputs": {"answer": answer},
                "metadata": {
                    "drug_name": main_key
                },  # Metadata includes the main key (drug name)
            }
            for question, answer in zip(inputs, outputs)
        ]
    )

try:
    # Create the combined dataset
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=combined_dataset_description,
    )

    # Add all examples to the dataset
    client.create_examples(
        inputs=[
            {"question": example["inputs"], "drugname": example["metadata"]}
            for example in examples
        ],
        outputs=[example["outputs"] for example in examples],
        dataset_id=dataset.id,
        metadata=[example["metadata"] for example in examples],  # Pass metadata
    )

    print(
        f"Dataset '{dataset_name}' created successfully with {len(examples)} examples."
    )
except Exception as e:
    print(f"Error creating dataset '{dataset_name}': {e}")


def get_answer(example):
    print(str(example))
    question = example["question"]["question"]
    drug_name = example["drugname"]["drug_name"]  # now accessible
    medication = DATA_TRANSFORM[drug_name][0]
    strength = DATA_TRANSFORM[drug_name][1]

    answer = present_result_filtered(question, medication, strength)
    print("----->" + str(answer))
    return answer


llm = OllamaLLM(
    model="llama3.3",
    base_url="https://gecadllm.fish-albacore.ts.net:8443/api",
    temperature=0,
    request_timeout=150,
)
llm_small = OllamaLLM(
    model="llama3.1",
    base_url="http://localhost:11434",
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
            "drug_name": example.metadata[
                "drug_name"
            ],  # Include drug name from metadata
        },
    )
]

experiment_results = evaluate(
    get_answer,
    data=dataset_name,
    evaluators=qa_evaluator,
    experiment_prefix="rag-qa-llama-nomic",
    metadata={"variant": "LCEL context, llama3.3 - nomic"},
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
