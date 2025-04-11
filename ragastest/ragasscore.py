

import os
import pandas as pd

from ragas import SingleTurnSample, evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.metrics import (
    LLMContextPrecisionWithReference,
#    LLMContextPrecisionWithoutReference,
#    NonLLMContextPrecisionWithReference,
    LLMContextRecall,
#    NonLLMContextRecall,
#    FactualCorrectness,
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

from ragastest.ragassupp import present_result
from ragastest.ragassupp import present_result_melhorado

from dotenv import load_dotenv

from datasets import Dataset
import ast
import re
#from ragas.run_config import RunConfig

load_dotenv()

#LLM_URL = os.getenv("LLM_URL")

#localModel = "llama3:8b"
#localEmbedding = "nomic-embed-text:latest"

#llm = LangchainLLMWrapper(OllamaLLM(model=localModel))
#llm = LangchainLLMWrapper(OllamaLLM(model="llama3.1:70b",base_url=LLM_URL))
#embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=localEmbedding))


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#OPEN_API_KEY = os.getenv("OPEN_API_KEY")

#llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
#embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

#run_conf = RunConfig(max_workers=5, timeout=100)

#def clean_text(text):
#    text_clean = text.replace('\r\n', ' ').strip()  # Remover quebras de linha
#    text_clean = re.sub(r"[•\[\]']", "", text_clean)
#    text_clean = re.sub(r'[•#\*\uf0b7]', '', text_clean) # Remove os caracteres '#' e '•'
#    text_clean= re.sub(r'\s+', ' ', text_clean)  # Remover múltiplos espaços
#        
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

#        context = [clean_text(text) for text in context]

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

    print("Resposta gerada!\nA avaliar a resposta...")

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

    df = pd.DataFrame(list(results.items()), columns=["Métrica", "Resultado"])
    return df



def evaluate_row(row, metrics):
    """
    Creates a single turn sample for the row
    Evaluates all metrics for it
    Returns a dictionary containing all metrics

    """

    # Create a SingleTurnSample for every row
    sample = SingleTurnSample(
        user_input = row['question'],
        reference = row['ground_truth'],
        retrieved_contexts = row['contexts'],
        response = row['answer']
    )
    
    # Evaluate metrics for the sample
    results = {}
    for metric_name, metric in metrics.items():
        try:
            #res = float(metric.single_turn_score(sample))
            #print(res)
            results[metric_name] = float(metric.single_turn_score(sample))            
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
    df = df.copy()

    df["contexts"] = df["contexts"].apply(ast.literal_eval)

    results = []
    for _, row in df.iterrows():
        row_results = evaluate_row(row, metrics)
        results.append(row_results)
    
    result = pd.DataFrame(results)
    
    medias = result.mean().round(3)
    result.loc["Mean"] = medias
    
    return result


#def use_evaluate_dataset(df, metrics):
#    """
#    Esta função, é uma outra abordagem usando o ragas e embora não
#    seja igual a evaluate_dataframe apresenta resultados
#    praticamente iguais. Para além disso tem uma complexidade maior porque 
#    diferentes valores nos parâmetros de run_config apresenta diferentes 
#    valores no score.
#    Embora ache que utilizar a função evaluate_dataframe seja melhor pelas 
#    razões acima, deixo esta função comentada caso queira experimentar.
#
#
#    """
#
#
#    df = df.copy()
#
#    df["contexts"] = df["contexts"].apply(ast.literal_eval)
#
#    questions = []
#    ground_truths = []
#    answers = []
#    contexts = []
#
#
#    for _, row in df.iterrows():
#
##        clean_context = [clean_text(text) for text in row['contexts']]
##        clean_question = clean_text(row['question'])
##        clean_ground_truth = clean_text(row['ground_truth'])
##        clean_answer = clean_text(row['answer'])
#
#        questions.append(row['question'])
#        ground_truths.append(row['ground_truth'])
#        contexts.append(row['contexts'])
#        answers.append(row['answer'])
#
#
#    # Define dataset
#    data_samples = {
#        'question': questions,
#        'answer': answers,
#        'contexts': contexts,
#        'ground_truth': ground_truths
#    }
#    data_set = Dataset.from_dict(data_samples)
#
#
#    score = evaluate(data_set, metrics=metrics, batch_size=4, llm = LangchainLLMWrapper(llm,run_config=run_conf), embeddings=LangchainEmbeddingsWrapper(embeddings, run_config=run_conf), run_config=run_conf)
#
#    dataf = score.to_pandas()
#    float_cols = dataf.select_dtypes(include=['float64', 'float32']).columns
#    mean_values = dataf[float_cols].mean().round(3)
#    mean_row = {col: '--' if dataf[col].dtype == 'object' else mean_values[col] for col in dataf.columns}
#    dataf.loc["Mean"] = mean_row
#
#
#
#    return dataf

eval_llm = LangchainLLMWrapper(llm)
eval_embedding = LangchainEmbeddingsWrapper(embeddings)
                               

metrics = {
            "Context Precision With Reference": LLMContextPrecisionWithReference(llm=eval_llm),
#            "Context Precision Without Reference": LLMContextPrecisionWithoutReference(llm=llm),
#            "Non LLM Context Precision With Reference": NonLLMContextPrecisionWithReference(),
            "Context Recall": LLMContextRecall(llm=eval_llm),
#            "Non LLM Context Recall": NonLLMContextRecall(),
            "Context Entities Recall": ContextEntityRecall(llm=eval_llm),
            "Noise Sensitivity": NoiseSensitivity(llm=eval_llm),
            "Response Relevancy": ResponseRelevancy(llm=eval_llm,embeddings=eval_embedding),
            "Faithfulness": Faithfulness(llm=eval_llm),
        }
## Estão três métricas comentadas pq nos exemplos que vi não eram usadas 
## para avaliar sistmas RAG, mas caso considere relevante também podem ser usadas
 

### Exemplos

#Avaliar a resposta de uma determinada pergunta
#per_res = pd.read_csv("ragastest/Perguntas e resposta.csv", delimiter=",")
#result = eval_sample(per_res["Pergunta"][2], per_res["Resposta"][2],metrics = metrics)
#print(result)


#Avaliar as respostas de um conjunto de perguntas guardadas num csv
#create_csv("dataset",per_res[3:7])
dataset = pd.read_csv("ragastest/dataset.csv", delimiter=",")
df_evaluation = evaluate_dataframe(df=dataset,metrics = metrics)
print(df_evaluation)




## Usando use_evaluate_dataset

#metrics1 = [
#            LLMContextPrecisionWithReference(), LLMContextRecall(), ContextEntityRecall(), NoiseSensitivity(), ResponseRelevancy(),Faithfulness(),
#        ]

#df_evaluation1 = use_evaluate_dataset(df=dataset,metrics = metrics1)
#print(df_evaluation1)



