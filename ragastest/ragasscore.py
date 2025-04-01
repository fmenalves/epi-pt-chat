

import os
import pandas as pd

from ragas import SingleTurnSample, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.metrics import (
    LLMContextPrecisionWithReference,
#    LLMContextPrecisionWithoutReference,
#    NonLLMContextPrecisionWithReference,
    LLMContextRecall,
#    NonLLMContextRecall,
    FactualCorrectness,
    ContextEntityRecall,
    NoiseSensitivity,
    ResponseRelevancy,
    Faithfulness,
)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
#from langchain_ollama.llms import OllamaLLM
#from langchain_community.embeddings import OllamaEmbeddings
#from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM

from ragassupp import present_result
from ragassupp import present_result_melhorado

from dotenv import load_dotenv

from datasets import Dataset
import ast
import re

load_dotenv()

#LLM_URL = os.getenv("LLM_URL")

localModel = "llama3:8b"
localEmbedding = "nomic-embed-text:latest"

llm = LangchainLLMWrapper(OllamaLLM(model=localModel))
#llm = LangchainLLMWrapper(Ollama(model="llama3.1:70b",base_url=LLM_URL))
embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=localEmbedding))



#GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

#llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model="gemini-2.0-flash"))
#embeddings = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(model="models/embedding-001"))



#OPEN_API_KEY = os.getenv("OPEN_API_KEY")

#embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
#llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))


#def clean_context(text):
#    text_clean = text.replace('\r\n', ' ').strip()  # Remover quebras de linha
#    text_clean= re.sub(r'\s+', ' ', text_clean)  # Remover múltiplos espaços
#    return text_clean



def create_csv(name, per_res):
    """
    Creates a CSV file containing questions, generated answers, 
    contexts, and ground truth answers.
   
    """

    questions = per_res["Pergunta"].tolist()

    ground_truth = per_res["Resposta"].tolist()

    data = {"question": [], "answer": [], "contexts": [], "ground_truth": ground_truth}

#    questions_melhoradas=[]


    for pergunta in questions:
        
        
#        resposta = present_result(pergunta)
        resposta = present_result_melhorado(pergunta)

        answer = resposta["response"]
        context = resposta["context"]

#        context = [clean_context(text) for text in context]

        data["question"].append(pergunta)

        data["answer"].append(answer)

        data["contexts"].append(context)

#        questions_melhoradas.append(resposta["nquery"])
    
#    questions = questions_melhoradas
    
    dataset = Dataset.from_dict(data)

    df = pd.DataFrame(dataset)

    df.to_csv(name+".csv", index=False, encoding="utf-8")



def eval_sample(Pergunta, ground_truth, metrics):
    """
    Evaluates the response generated for a given question
    Retrieves the response and contexts
    Evaluates all metrics for it
    Returns a dictionary containing all metrics

    """


    resposta = present_result_melhorado(Pergunta)
#    resposta = present_result(Pergunta)

    sample = SingleTurnSample(
    user_input=Pergunta,
    reference=ground_truth,
    retrieved_contexts=resposta["context"],
    response = resposta["response"]
    )

    # Defining each metric that we wanna see
    results = {}
    # Iterating through the metrics dictionary
    for metric_name, metric in metrics.items():

        try:
            results[metric_name] = metric.single_turn_score(sample)

        except Exception as e:
            results[metric_name] = f"Error: {e}"

    return results



def evaluate_row(row, metrics):
    """
    Creates a single turn sample for the row
    Evaluates all metrics for it
    Returns a dictionary containing all metrics

    """

    # Create a SingleTurnSample for every row
    sample = SingleTurnSample(
        user_input=row['question'],
        reference=row['ground_truth'],
        retrieved_contexts=row['contexts'],
        response = row['answer']
    )
    
    # Evaluate metrics for the sample
    results = {}
    for metric_name, metric in metrics.items():
        try:
            results[metric_name] = metric.single_turn_score(sample)
        except Exception as e:
            results[metric_name] = f"Error: {e}"
    
    return results

# Function to evaluate the entire dataframe of testset

def evaluate_dataframe(df, metrics):
    """
    Iterates through the df test set
    For every row uses evaluate_row function to get result dictionary
    Append each dictionary to a list
    Uses list to create the result dataframe

    """

    df["contexts"] = df["contexts"].apply(ast.literal_eval)

    results = []
    for _, row in df.iterrows():
        row_results = evaluate_row(row, metrics)
        results.append(row_results)
    
    res = pd.DataFrame(results)
    
    medias = res.mean().round(3)
    res.loc["Mean"] = medias
    
    return res


metrics = {
            "Context Precision With Reference": LLMContextPrecisionWithReference(llm=llm),
#            "Context Precision Without Reference": LLMContextPrecisionWithoutReference(llm=llm),
#            "Non LLM Context Precision With Reference": NonLLMContextPrecisionWithReference(),
            "Context Recall": LLMContextRecall(llm=llm),
#            "Non LLM Context Recall": NonLLMContextRecall(),
            "Context Entities Recall": ContextEntityRecall(llm=llm),
            "Noise Sensitivity": NoiseSensitivity(llm=llm),
#            "Response Relevancy": ResponseRelevancy(llm=llm,embeddings=embeddings),
            "Faithfulness": Faithfulness(llm=llm),
        }

 


# Exemplo

#per_res = pd.read_csv("Perguntas e resposta.csv", delimiter=",")
#result = eval_sample(per_res["Pergunta"][0], per_res["Resposta"][0],metrics = metrics)



#create_csv("dataset",per_res[3:7])
dataset = pd.read_csv("dataset.csv", delimiter=",")
df_evaluation = evaluate_dataframe(dataset,metrics = metrics)


#test3 = pd.read_csv("test3.csv", delimiter=",")

#df_evaluation = evaluate_dataframe(test3,metrics = metrics) 


