###TODO

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
from llama_index.embeddings.ollama import OllamaEmbedding
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

import re

from app.support import (
    get_filters_qdrant,
    get_filters_qdrant_filtered
)


# Configuração inicial
load_dotenv()

# Configurações de ambiente
LLM_URL = os.getenv("LLM_URL")
URI_BD = os.getenv("URI_BD")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

# Inicialização do cliente Qdrant
#client = qdrant_client.QdrantClient(URI_BD)

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
#metadatasource = pd.read_csv("finaldbpt2.csv", delimiter=",")

# Templates de prompts

# Template para extrair nomes de medicamentos e substâncias
EXTRACT_MEDS = PromptTemplate(

    "You are an information extraction agent specialized in identifying drug names.\n\n"
    "Instructions:\n"
    "1. Extract all commercial drug names and/or active substances mentioned in the input text.\n"
    "2. If only commercial drug names are mentioned, include their corresponding active substances in the response, separated by commas.\n"
    "3. **Return only the names, separated by commas. Do not include any punctuation at the end, explanations, or extra formatting.**\n"
    "4. If no commercial drug names or active substances are mentioned, return a single dash: '-'\n"
    "5. Capitalize names correctly.\n\n"
    "Input: {query}"            

)

# Template para fornecer informações adicionais sobre medicamentos
MED_INFO = PromptTemplate(

    "You are a clinical pharmacist assistant. Your task is to provide structured and accurate drug information.\n\n"
    "Instructions:\n"
    "1. For each commercial drug or active substance listed, provide detailed information about: Active substance, Common side effects, Drug class, Known interactions with other drugs, Typical dosage (if applicable) and Contraindications\n."
    "2. Organize the output by drug or substance.\n"
    "3. Always provide information in **European Portuguese**, avoiding vocabulary or expressions specific to Brazilian Portuguese.\n"
    "4. Use clear, medically accurate language.\n"
    "5. Do not include any information unrelated to the listed products.\n\n"
    "Input:\n"
    "Products: {products}"

)

# Template para seleção de contextos
CONTEXT_SELECTION = PromptTemplate(

    "The fragments information is"
    " below.\n---------------------\n{fragments}\n---------------------\nUsing"
    " the fragments information select the {top_n} fragments that are most relevant to"
    " the question: {query}\n"
    "Return only the indices of the selected fragments, separated by"
    " commas. Do not include any punctuation at the end, explanations, or extra formatting.\n"
    "Answer:\n"
    
)

# Template para QA com contexto
TEXT_QA = PromptTemplate(

    "You are a question-answering assistant specialized in processing context documents.\n\n"
    "Instructions:\n"
    "1. Use **only** the context provided to answer the question and not prior knowledge.\n"
    "2. If the context isn't helpful, you may use your own knowledge to respond.\n"
    "3. Always answer in **European Portuguese**, avoiding vocabulary or expressions specific to Brazilian Portuguese.\n"
    "4. If the answer is based on any part of the context, cite the source (e.g., document title or reference).\n"
    "5. Do not refer to the context as 'the context' in your answer — integrate the information naturally.\n"
    "6. Be clear, accurate, and informative.\n\n"
    "Input:\n"
    "Context information is:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Question: {query_str}\n\n"
    "Answer:"

)

# Template para refinamento de respostas 
REFINE = PromptTemplate(

    "You are an assistant responsible for refining existing answers.\n\n"
    "Instructions:\n"
    "1. Read the original question and the existing answer.\n"
    "2. Analyze the additional context provided below the separator.\n"
    "3. If the context contains relevant and useful information, update or improve the existing answer accordingly.\n"
    "4. If the context does not add value or relevant information, keep the original answer unchanged.\n"
    "5. Use **only** the context provided — do not use prior knowledge.\n"
    "6. Always write in **European Portuguese**, avoiding expressions or vocabulary from Brazilian Portuguese.\n"
    "7. If you incorporate new information from the context, **cite the source** clearly.\n"
    "8. Ensure your response is clear, accurate, and naturally written.\n\n"
    "Input:\n"
    "- Original question: {query_str}\n"
    "- Existing answer: {existing_answer}\n"
    "New context:\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n\n"
    "Refined Answer:"

)

# Template para reflexão sobre a qualidade da resposta
REFLECTION = PromptTemplate(

    "You are an expert evaluator of medical information responses.\n\n"
    "Instructions:\n"
    "1. Analyze critically the provided answer to a medication-related question.\n"
    "2. Identify and list any issues related to:\n"
    "- Incorrect or inaccurate information\n"
    "- Missing relevant data\n"
    "- Inconsistencies or contradictions\n"
    "- Use of Brazilian Portuguese instead of European Portuguese\n"
    "- Lack of source citations\n"
    "3. Provide a clear evaluation of the answer.\n"
    "4. Suggest specific improvements to enhance the accuracy, completeness, language, and referencing of the response.\n\n"
    "Input:\n"
    "- Question: {query}\n"
    "- Answer: {response}\n\n"
    "Provide your evaluation and improvement suggestions below:"

)
    
# Template para integrar informações e criar uma resposta final
INTEGRATE = PromptTemplate(

    "You are a pharmaceutical consultant assistant tasked with integrating information from multiple sources to answer medication-related queries.\n\n"
    "Instructions:\n"
    "1. Use the information retrieved from the database, additional medication details, and reflections on potential issues to create a complete and well-structured response.\n"
    "2. Address the original question clearly and precisely.\n"
    "3. Organize your answer logically, covering all the relevant aspects including:\n"
    "4. Cite sources where appropriate.\n"
    "5. Write the response in European Portuguese, ensuring clarity and completeness.\n\n"
    "Input:\n"
    "- Original question: {query}\n"
    "- Retrieved information: {retrieved_info}\n"
    "- Additional medication information: {med_info}\n"
    "- Reflections on potential issues: {reflection}\n\n"
    "Provide the comprehensive answer below:"

)




def to_single_line(text: str, max_chars: int | None = None) -> str:
    """
    Converte o conteúdo de um nó para UMA única linha legível.
    - Desfaz hifenização em quebra de linha
    - Remove linhas que são só números (artefactos de página)
    - Normaliza bullets
    - Converte quebras de linha em espaços e compacta múltiplos espaços
    - (Opcional) trunca a N caracteres
    """
    if not isinstance(text, str):
        text = str(text)
    t = text.replace("\r", "")

    # desfaz hifenização em quebra de linha: palavra-\nseguinte -> palavra seguinte
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1 \2", t)

    # remove linhas que são só números (nºs de página)
    t = re.sub(r"^\s*\d{1,3}\s*$", "", t, flags=re.M)

    # normaliza bullets
    t = t.replace("", "- ").replace("•", "- ")

    # converte quebras de linha para espaços
    t = re.sub(r"\s*\n\s*", " ", t)

    # compacta espaços
    t = re.sub(r"[ \t]{2,}", " ", t).strip()

    # truncagem opcional
    if max_chars is not None and len(t) > max_chars:
        t = t[:max_chars].rstrip() + " …"

    return t


class ContextSelectorPostProcessor(BaseNodePostprocessor):
    """
    Processador de pós-recuperação que utiliza um prompt template para selecionar
    os contextos mais relevantes para uma consulta sobre medicamentos.
    """
    
    def __init__(self, llm, prompt_template, top_n: int = 15):
        """
        Inicializa o seletor de contextos.
        
        Args:
            llm: O modelo de linguagem a ser utilizado
            prompt_template: O template de prompt para avaliação de contextos
            top_n: Número de contextos a retornar após classificação
        """
        super().__init__()
        self._llm = llm
        self._prompt_template = prompt_template
        self._top_n = top_n
        
    
#    def _postprocess_nodes(self, nodes: List[NodeWithScore], query_str: str) -> List[NodeWithScore]:
    def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Avalia e classifica os nós de contexto com base na sua relevância para a consulta.
        
        Args:
            nodes: Lista de nós recuperados pelo retriever
            query_str: A consulta original do usuário
            
        Returns:
            Lista dos nós mais relevantes, limitados a top_n
        """
        #ver questão QueryBundle, se não usar apagar linha abaixo
        query_str = query_bundle.query_str

        if not nodes or len(nodes) <= self._top_n:
            return nodes
            
        # Extrair textos dos nós para avaliação
        fragments_raw = {i: node.text for i, node in enumerate(nodes)}
        
        fragments_one_line = {i: to_single_line(txt, max_chars=None)  # ajusta max_chars se quiseres
                      for i, txt in fragments_raw.items()}
        
        
        fragments_block = "\n".join(
            f"FRAGMENTO {i}: {txt}" for i, txt in sorted(fragments_one_line.items())
)

        print(fragments_block)

        
        
        try:
            # Obter classificação do LLM usando o template de prompt
            response = self._llm.predict(
                self._prompt_template,
                query=query_str,
                fragments=fragments_block,
                top_n=self._top_n
            )

            print("Resposta do LLM para seleção de contextos:")
            print(response)
            
            # Processar a resposta para obter os índices
            selected_indices = []
            for part in response.strip().replace(" ", "").split(','):
                try:
                    idx = int(part.strip())
                    if 0 <= idx < len(nodes):
                        selected_indices.append(idx)
                except ValueError:
                    continue
            
            #print(selected_indices)
            
            # Limitar ao número desejado
            selected_indices = selected_indices[:self._top_n]

            print(selected_indices)
            
            # Se não conseguirmos extrair índices válidos, use os primeiros top_n
            if not selected_indices:
                return nodes[:self._top_n]
            
            #print([nodes[idx] for idx in selected_indices])
            
            # Retornar os nós selecionados na ordem de relevância
            return [nodes[idx] for idx in selected_indices]
            
        except Exception as e:
            print(f"Erro na seleção de contextos: {e}")
            # Fallback para os primeiros top_n nós
            return nodes[:self._top_n]



# Classe para índice de recuperação

"""Inicializa o índice de recuperação"""
# Configurar divisor de texto


def retrieve_index(client, llm, index_name):
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
    Settings.text_splitter = text_splitter

    ## embed_model = OpenAIEmbedding(embed_batch_size=10)
    #embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    embed_model = OllamaEmbedding(model_name="nomic-embed-text")
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
    
def create_query_engine(client, filters_qdrant, ret_similarity_top_k, rer_top_n, Cohere):
    """Cria um motor de consulta com base nos filtros fornecidos"""
    
    index = retrieve_index(client, llm, INDEX_NAME)
    
    retriever = VectorIndexRetriever(
        vector_store_kwargs={"qdrant_filters": filters_qdrant},
        index=index,
        similarity_top_k=ret_similarity_top_k
    )

    if Cohere:

        reranker = CohereRerank(api_key=COHERE_API_KEY, top_n=rer_top_n, model="rerank-v3.5") # model = "rerank-english-v3.0"
    
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[reranker],
        )

    else:

        # Criar seletor de contextos como pós-processador
        context_selector = ContextSelectorPostProcessor(
            llm=llm,
            prompt_template =CONTEXT_SELECTION,
            top_n=rer_top_n  # Limitar aos 15 melhores contextos
        )
    
        # Montar motor de consulta
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[context_selector]
        #    node_postprocessors=[reranker],
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

def get_medication_info(products):
    """Obtém informações detalhadas sobre medicamentos"""
    med_info = llm.predict(MED_INFO, products=products)
    print(med_info)
    return med_info

def retrieve_information(query, client, products, strength, metadatasource, enhance_query, ret_similarity_top_k, rer_top_n, Cohere):
    """Recupera informações da base de conhecimento"""
    # Criar filtros para consulta
    if strength:  # demo
        filters_qdrant = get_filters_qdrant_filtered(products, strength)
    else:  # not demo
        filters_qdrant = get_filters_qdrant(metadatasource = metadatasource, products = products)
    
    # Criar motor de consulta
    query_engine = create_query_engine(client, filters_qdrant, ret_similarity_top_k, rer_top_n, Cohere)
    
    # Enriquecer consulta com contexto adicional
    med_info = get_medication_info(products)

    if enhance_query:
        query = f"{query}\n---------\nContexto e mais informações sobre os produtos:\n{med_info}"
    
    # Executar consulta
    result = query_engine.query(query)
    
    # Extrair contextos
    contexts = [node.text for node in result.source_nodes]

    print("Response from query engine:")
    print(result.response)
    
    return {
        "response": result.response,
        "contexts": contexts,
        "query": query,
        "med_info": med_info
    }

def evaluate_response(query, response):
    """Avalia a qualidade da resposta"""
    reflection = llm.predict(REFLECTION, query=query, response=response)
    print("Reflection on Response:")
    print(reflection)
    
    return reflection

def integrate_information(query, retrieved_info, med_info, reflection):
    """Integra todas as informações em uma resposta final"""
    integrate = llm.predict(
        INTEGRATE,
        query=query,
        retrieved_info=retrieved_info,
        med_info=med_info,
        reflection=reflection
    )
    print("Integrated Response:")
    print(integrate)
    
    return integrate

def agent_process_query(query: str, client, metadatasource, products: str = None, strength: str = None, enhance_query: bool = False,
                       ret_similarity_top_k: int = 10, rer_top_n: int = 2, Cohere: bool = False) -> dict[str, object]:
    """Processa uma consulta completa através do sistema AgenticRAG"""
    start = timeit.default_timer()
    
    
    # 1. Extrair medicamentos mencionados
    if products == None:
        products = extract_medications(query)
        print(f"Medicamentos detectados: {products}")
    
    # 2. Recuperar informações relevantes da base de conhecimento
    retrieval_result = retrieve_information(query, client, products, strength, metadatasource, enhance_query, ret_similarity_top_k, rer_top_n, Cohere)
    
    # 3. Avaliar a resposta inicial
    reflection = evaluate_response(query, retrieval_result["response"])
    
    # 4. Integrar todas as informações em uma resposta final
    final_response = integrate_information(
        query,
        retrieval_result["response"],
        retrieval_result["med_info"],
        reflection
    )
    
    end = timeit.default_timer()
    execution_time = round(end - start)
    
    # Preparar resultado final
    result = {
        "query": query,
        "products_detected": products,
        "med_info": retrieval_result["med_info"],
        "response": final_response,
        "contexts": retrieval_result["contexts"],
        "execution_time": f"{execution_time}s"
    }
    
    return result