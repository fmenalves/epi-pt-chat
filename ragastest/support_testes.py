

import os
import json
import timeit
from dotenv import load_dotenv
import pandas as pd

# Componentes de vetorização e embeddings
from llama_index.core import Settings, VectorStoreIndex, PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny

# Componentes de recuperação e processamento
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.query_engine import RetrieverQueryEngine

# Componentes para LLM
from llama_index.llms.ollama import Ollama

from typing import List
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

# Configuração inicial
load_dotenv()

# Configurações de ambiente
LLM_URL = os.getenv("LLM_URL")
URI_BD = os.getenv("URI_BD")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

# Inicialização do cliente Qdrant
client = qdrant_client.QdrantClient(URI_BD)

# Inicialização do modelo LLM
llm = Ollama(
    model="llama3.1:70b",  
    base_url=LLM_URL,
    temperature=0,
    request_timeout=120,
)

# Modelo de embeddings
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Carregamento dos metadados dos medicamentos
metadatasource = pd.read_csv("finaldbpt2.csv", delimiter=",")

# Templates de prompts

# Template para extrair nomes de medicamentos e substâncias
EXTRACT_MEDS = PromptTemplate(
    
    """You are a helpful assistant that can provide information about drugs. Please comercial drug names and/or substances, separated by a comma. You will return only the names separeted by comma and nothing more. The text to analyse is {query}."""
)

MED_INFO = PromptTemplate(
    
    """You are a helpful assistant that can provide information about drugs. Please provide information additional about the drugs detected in the query. Important things to mention are active principles, side effects, drug class and interactions with other drugs for each detected drug.
    Query: {query}
    """
)

TEXT_QA = PromptTemplate(
    "Context information is"
    " below.\n---------------------\n{context_str}\n---------------------\nUsing"
    " the context information and not prior knowledge, answer"
    " the question: {query_str}\nIf the context isn't helpful, you can also"
    " answer the question on your own."
    "Respond always in the language portuguese from portugal and do not use any word that is specific from brazil."
    "Cite the source of the new information if you use it.\n"
    "Answer:\n"

)

REFINE = PromptTemplate(
    "The original question is as follows: {query_str}\nWe have provided an"
    " existing answer: {existing_answer}\nWe have the opportunity to refine"
    " the existing answer (only if needed) with some more context"
    " below.\n------------\n{context_msg}\n------------\nUsing the"
    " context information and not prior knowledge, update or repeat the existing answer.\n"
    " Respond always in the language portuguese from portugal and do not use any word that is specific from brazil.\n"
    "Cite the source of the new information if you use it.\n"
    "If the context isn't useful, return the original answer.\n"
    "Refined Answer: "
)






        
def create_base_filters(products, metadatasource):
    """Cria filtros básicos para nomes comerciais e substâncias"""
    filters = {"Nome_Comercial": [], "Substancia": []}
    
    for word in products.split(","):
        word = word.strip()
        
        # Correspondência exata para Nome Comercial
        if any(word.lower() == item.lower() for item in metadatasource["Nome Comercial"]):
            val = metadatasource[
                [word.lower() == item.lower() for item in metadatasource["Nome Comercial"]]
            ]["Nome Comercial"].values
            
            for v in val:
                filters["Nome_Comercial"].append(v)
        
        # Correspondência exata para Substância
        elif any(word.lower() == str(item).lower() for item in metadatasource["Subs"]):
            val = metadatasource[
                [word.lower() == str(item).lower() for item in metadatasource["Subs"]]
            ]["Subs"].values
            
            for v in val:
                filters["Substancia"].append(v)
        
        # Correspondência parcial para Nome Comercial
        if metadatasource["Nome Comercial"].str.contains(word, case=False).any():
            val = metadatasource[
                metadatasource["Nome Comercial"].str.contains(word, case=False)
            ]["Nome Comercial"].values
            
            for v in val:
                filters["Nome_Comercial"].append(v)
        
        # Correspondência parcial para Substância
        elif (
            metadatasource[metadatasource["Subs"].notna()]["Subs"]
            .str.contains(word, case=False)
            .any()
        ):
            val = metadatasource[
                metadatasource["Subs"].notna()
                & metadatasource["Subs"].str.contains(word, case=False)
            ]["Subs"].values
            
            for v in val:
                filters["Substancia"].append(v)
    
    return filters

def get_qdrant_filters(metadatasource, products):
    """Converte filtros básicos para o formato aceito pelo Qdrant"""
    filters = []
    base_filters = create_base_filters(products=products, metadatasource=metadatasource)
    
    if len(base_filters["Nome_Comercial"]) > 0:
        filters.append(
            FieldCondition(
                key="Nome_Comercial",
                match=MatchAny(any=list(set(base_filters["Nome_Comercial"]))),
            )
        )
    
    if len(base_filters["Substancia"]) > 0:
        filters.append(
            FieldCondition(
                key="Substancia",
                match=MatchAny(any=list(set(base_filters["Substancia"]))),
            )
        )
    
    if len(filters) == 0:
        return Filter(should=[])
    
    return Filter(should=filters)

def get_filters_qdrant_filtered(products, strength):
    filters = []
    print(products, strength)
    filters.append(
        FieldCondition(key="Nome_Comercial", match=MatchAny(any=[products])),
    )

    filters.append(
        FieldCondition(
            key="Dosagem",
            match=MatchAny(any=[strength]),
        )
    )
    if len(filters) == 0:
        return Filter(should=[])
    if len(filters) > 0:
        return Filter(must=filters)



def retrieve_index(client, llm, index_name):
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
    Settings.text_splitter = text_splitter

    ## embed_model = OpenAIEmbedding(embed_batch_size=10)
    embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    Settings.llm = llm
    Settings.embed_model = embed_model
    ## Settings.num_output = 512
    ## Settings.context_window = 3900
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 64

    vector_store = QdrantVectorStore(client=client, collection_name=index_name)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        text_qa_template=TEXT_QA,
        refine_template=REFINE,
        transformations=[text_splitter],
    )

    return index
    
def create_query_engine(filters_qdrant, ret_similarity_top_k, rer_top_n):
    
    index = retrieve_index(client, llm, INDEX_NAME)
    
    retriever = VectorIndexRetriever(
        vector_store_kwargs={"qdrant_filters": filters_qdrant},
        index=index,
        similarity_top_k=ret_similarity_top_k
    )

    reranker = CohereRerank(api_key=COHERE_API_KEY, top_n=rer_top_n)
    
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[reranker],
    )
    
    # Atualizar prompts do motor
    query_engine.update_prompts(
        {
            "response_synthesizer:text_qa_template": TEXT_QA,
            "response_synthesizer:refine_template": REFINE,
        }
    )
    
    return query_engine


# Funções do sistema AgenticRAG Simplificado


def extract_medications(query):
    """Extrai medicamentos da consulta"""
    response = llm.predict(EXTRACT_MEDS, query=query)
    return response

def get_medication_info(query):
    """Obtém informações detalhadas sobre medicamentos"""
    return llm.predict(MED_INFO, query=query)

def retrieve_information(query, products, strength, enhance_query, ret_similarity_top_k, rer_top_n):
    """Recupera informações da base de conhecimento"""
    # Criar filtros para consulta
    if strength:  # demo
        filters_qdrant = get_filters_qdrant_filtered(products, strength)
    else:  # not demo
        filters_qdrant = get_qdrant_filters(products)
    
    # Criar motor de consulta
    query_engine = create_query_engine(filters_qdrant, ret_similarity_top_k=ret_similarity_top_k, rer_top_n=rer_top_n)
    
    # Enriquecer consulta com contexto adicional
    med_info = get_medication_info(query)

    if enhance_query:
        query = f"{query}\n---------\nContexto e mais informações sobre os produtos:\n{med_info}"
    
    # Executar consulta
    result = query_engine.query(query)
    
    # Extrair contextos
    contexts = [node.text for node in result.source_nodes]
    
    return {
        "response": result.response,
        "contexts": contexts,
        "query": query,
        "med_info": med_info
    }


def process_query(query: str, products: str = None, strength: str = None, enhance_query: bool = False,
                       ret_similarity_top_k: int = 30, rer_top_n: int = 15) -> dict[str, object]:
    """Processa uma consulta completa através do sistema AgenticRAG"""
    start = timeit.default_timer()
    
    
    # 1. Extrair medicamentos mencionados
    if products == None:
        products = extract_medications(query)
        print(f"Medicamentos detectados: {products}")
    
    # 2. Recuperar informações relevantes da base de conhecimento
    retrieval_result = retrieve_information(query, products, strength, enhance_query, ret_similarity_top_k, rer_top_n)
    
    end = timeit.default_timer()
    execution_time = round(end - start)
    
    # Preparar resultado final
    result = {
        "query": query,
        "products_detected": products,
        "med_info": retrieval_result["med_info"],
        "response": retrieval_result["response"],
        "contexts": retrieval_result["contexts"],
        "execution_time": f"{execution_time}s"
    }
    
    return result



# Exemplo de uso
if __name__ == "__main__":
    # Exemplo de consulta
    query = "Quais cuidados devem ser tomados com pacientes acima de 65 anos ao usar RINVOQ?"
    
    print("Processando consulta...")
    result = process_query(query, "Rinvoq", "15 mg")
    
    print("\n=== RESULTADO ===")
    print(f"Consulta: {result['query']}")
    print(f"Medicamentos detectados: {result['products_detected']}")
    print(f"Tempo de execução: {result['execution_time']}")
    print("\nResposta:")
    print(result['response'])
    
    print("\n=== FONTES ===")
    for i, context in enumerate(result['contexts'][:3], 1):
        print(f"\nFonte {i}:")
        print(context[:200] + "..." if len(context) > 200 else context)