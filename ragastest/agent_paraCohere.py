

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
#COHERE_API_KEY = os.getenv("COHERE_API_KEY")
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
class PromptTemplates:
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

    # Template para QA com contexto
    TEXT_QA = PromptTemplate(
    
        "You are a question-answering assistant specialized in processing context documents.\n\n"
        "Instructions:\n"
        "1. Use **only** the context provided to answer the question and not prior knoledge.\n"
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


# Classe para memória do sistema
class QueryMemory:
    def __init__(self):
        self.query_history = []
        self.extracted_products = {}
        self.retrieved_contexts = {}
        self.reflections = {}
    
    def add_query(self, query: str) -> None:
        self.query_history.append(query)
    
    def add_products(self, query: str, products: List[str]) -> None:
        self.extracted_products[query] = products
    
    def add_contexts(self, query: str, contexts: List[str]) -> None:
        self.retrieved_contexts[query] = contexts
    
    def add_reflection(self, query: str, reflection: str) -> None:
        self.reflections[query] = reflection
    
    def get_recent_queries(self, limit: int = 3) -> List[str]:
        return self.query_history[-limit:] if self.query_history else []
    
    def get_products_for_query(self, query: str) -> List[str]:
        return self.extracted_products.get(query, [])
    
    def summarize(self) -> dict[str, object]:
        """Retorna um resumo da memória atual"""
        return {
            "total_queries": len(self.query_history),
            "recent_queries": self.get_recent_queries(),
            "extracted_products_count": {q: len(p) for q, p in self.extracted_products.items()},
        }


# Classe para filtros personalizados
class MedicationFilter:
    def __init__(self, metadatasource: pd.DataFrame):
        self.metadatasource = metadatasource
        
    def create_base_filters(self, products: str) -> dict[str, list[str]]:
        """Cria filtros básicos para nomes comerciais e substâncias"""
        filters = {"Nome_Comercial": [], "Substancia": []}
        
        for word in products.split(","):
            word = word.strip()
            
            # Correspondência exata para Nome Comercial
            if any(word.lower() == item.lower() for item in self.metadatasource["Nome Comercial"]):
                val = self.metadatasource[
                    [word.lower() == item.lower() for item in self.metadatasource["Nome Comercial"]]
                ]["Nome Comercial"].values
                
                for v in val:
                    filters["Nome_Comercial"].append(v)
            
            # Correspondência exata para Substância
            elif any(word.lower() == str(item).lower() for item in self.metadatasource["Subs"]):
                val = self.metadatasource[
                    [word.lower() == str(item).lower() for item in self.metadatasource["Subs"]]
                ]["Subs"].values
                
                for v in val:
                    filters["Substancia"].append(v)
            
            # Correspondência parcial para Nome Comercial
            if self.metadatasource["Nome Comercial"].str.contains(word, case=False).any():
                val = self.metadatasource[
                    self.metadatasource["Nome Comercial"].str.contains(word, case=False)
                ]["Nome Comercial"].values
                
                for v in val:
                    filters["Nome_Comercial"].append(v)
            
            # Correspondência parcial para Substância
            elif (
                self.metadatasource[self.metadatasource["Subs"].notna()]["Subs"]
                .str.contains(word, case=False)
                .any()
            ):
                val = self.metadatasource[
                    self.metadatasource["Subs"].notna()
                    & self.metadatasource["Subs"].str.contains(word, case=False)
                ]["Subs"].values
                
                for v in val:
                    filters["Substancia"].append(v)
        
        return filters
    
    def get_qdrant_filters(self, products: str) -> Filter:
        """Converte filtros básicos para o formato aceito pelo Qdrant"""
        filters = []
        base_filters = self.create_base_filters(products=products)
        
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
    
    def get_filters_qdrant_filtered(self, products: str, strength: str) -> Filter:
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



# Classe para índice de recuperação
class MedicationIndexRetriever:
    def __init__(self, client, llm, index_name):
        self.client = client
        self.llm = llm
        self.index_name = index_name
        self.index = None
        self._initialize_index()
    
    def _initialize_index(self):
        """Inicializa o índice de recuperação"""
        # Configurar divisor de texto
        text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
        Settings.text_splitter = text_splitter
        
        # Configurar modelo de embedding
        embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        
        # Configurar definições globais
        Settings.llm = self.llm
        Settings.embed_model = embed_model
        Settings.chunk_size = 1024
        Settings.chunk_overlap = 64
        
        # Criar índice vetorial
        vector_store = QdrantVectorStore(client=self.client, collection_name=self.index_name)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store,
            text_qa_template=PromptTemplates.TEXT_QA,
            refine_template=PromptTemplates.REFINE,
            transformations=[text_splitter]
        )
    
    def create_query_engine(self, filters_qdrant: Filter, ret_similarity_top_k: int = 30, rer_top_n: int = 15):

        """Cria um motor de consulta com base nos filtros fornecidos"""
        retriever = VectorIndexRetriever(
            vector_store_kwargs={"qdrant_filters": filters_qdrant},
            index=self.index,
            similarity_top_k=ret_similarity_top_k
        )

        
        # Adicionar reranker para melhorar relevância
        #reranker = CohereRerank(api_key=COHERE_API_KEY, top_n=15)
        
        # Montar motor de consulta
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            #node_postprocessors=[reranker],
        )
        
        # Atualizar prompts do motor
        query_engine.update_prompts(
            {
                "response_synthesizer:text_qa_template": PromptTemplates.TEXT_QA,
                "response_synthesizer:refine_template": PromptTemplates.REFINE,
            }
        )
        
        return query_engine


# Funções do sistema AgenticRAG Simplificado
class AgenticRagSystem:
    def __init__(self):
        self.llm = llm
        self.filter_system = MedicationFilter(metadatasource)
        self.index_retriever = MedicationIndexRetriever(client, llm, INDEX_NAME)
        self.memory = QueryMemory()
    
    def extract_medications(self, query: str) -> str:
        """Extrai medicamentos da consulta"""
        response = self.llm.predict(PromptTemplates.EXTRACT_MEDS, query=query)
        # Armazenar na memória
        self.memory.add_products(query, response.split(","))
        return response
    
    def get_medication_info(self, products: str) -> str:
        """Obtém informações detalhadas sobre medicamentos"""
        return self.llm.predict(PromptTemplates.MED_INFO, products=products)
    
    def retrieve_information(self, query: str, products: str, strength: str, enhance_query: bool = False) -> dict[str, object]:
        """Recupera informações da base de conhecimento"""
        # Criar filtros para consulta
        if strength:  # demo
            filters_qdrant = self.filter_system.get_filters_qdrant_filtered(products=products, strength=strength)
        else:  # not demo
            filters_qdrant = self.filter_system.get_qdrant_filters(products=products)
        
        # Criar motor de consulta
        query_engine = self.index_retriever.create_query_engine(filters_qdrant=filters_qdrant)
        
        # Enriquecer consulta com contexto adicional
        med_info = self.get_medication_info(products)

        if enhance_query:
            query = f"{query}\n---------\nContexto e mais informações sobre os produtos:\n{med_info}"
        
        # Executar consulta
        result = query_engine.query(query)
        
        # Extrair contextos
        contexts = [node.text for node in result.source_nodes]
        
        # Armazenar na memória
        self.memory.add_contexts(query, contexts)
        
        return {
            "response": result.response,
            "contexts": contexts,
            "query": query,
            "med_info": med_info
        }
    
    def evaluate_response(self, query: str, response: str) -> str:
        """Avalia a qualidade da resposta"""
        reflection = self.llm.predict(PromptTemplates.REFLECTION, query=query, response=response)
        
        # Armazenar na memória
        self.memory.add_reflection(query, reflection)
        
        return reflection
    
    def integrate_information(self, query: str, retrieved_info: str, med_info: str, reflection: str) -> str:
        """Integra todas as informações em uma resposta final"""
        return self.llm.predict(
            PromptTemplates.INTEGRATE,
            query=query,
            retrieved_info=retrieved_info,
            med_info=med_info,
            reflection=reflection
        )
    
    def process_query(self, query: str, products: str, strength: str) -> dict[str, object]:
        """Processa uma consulta completa através do sistema AgenticRAG"""
        start = timeit.default_timer()
        
        # Adicionar consulta à memória
        self.memory.add_query(query)
        
        # 1. Extrair medicamentos mencionados
        if products == None:
            products = self.extract_medications(query)
            print(f"Medicamentos detectados: {products}")
        
        # 2. Recuperar informações relevantes da base de conhecimento
        retrieval_result = self.retrieve_information(query, products, strength)
        
        # 3. Avaliar a resposta inicial
        reflection = self.evaluate_response(query, retrieval_result["response"])
        
        # 4. Integrar todas as informações em uma resposta final
        final_response = self.integrate_information(
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


# Função principal para processar consultas
def answer_medication_query_agentic(query: str, products: str = None, strength: str = None) -> dict[str, object]:
    """Função principal para processar consultas de medicamentos"""
    system = AgenticRagSystem()
    return system.process_query(query=query, products=products, strength=strength)


# Exemplo de uso
if __name__ == "__main__":
    # Exemplo de consulta
    query = "Quais cuidados devem ser tomados com pacientes acima de 65 anos ao usar RINVOQ?"
    
    print("Processando consulta...")
    result = answer_medication_query_agentic(query, "Rinvoq", "15 mg")
    
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