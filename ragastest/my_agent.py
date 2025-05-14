"""
AgenticRAG Simplificado para Consulta de Medicamentos

Este script implementa um sistema de Retrieval-Augmented Generation (RAG) com capacidades agênticas
para responder a consultas sobre medicamentos em português, sem depender do CrewAI.

Componentes principais:
1. Funções especializadas para identificação de medicamentos e análise de informações
2. Sistema de memória para manter o contexto das consultas
3. Filtros dinâmicos personalizados para consulta de vetores
4. Sistema de reflexão para auto-avaliação e melhoria de respostas

Requer:
- Qdrant como base de dados vetorial (via Docker)
- Modelo LLM via Ollama (llama3.1:70b ou outro modelo disponível)
- Bibliotecas Python: llama-index, langchain, etc.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
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
    model="llama3.1:70b",  # Pode substituir por um modelo menor como "llama3.1:instruct"
    base_url=LLM_URL,
    temperature=0,
    request_timeout=120,
)

# Modelo de embeddings
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

# Carregamento dos metadados dos medicamentos
metadatasource = pd.read_csv("finaldbpt2.csv", delimiter=",")

# Templates de prompts
class PromptTemplates:
    # Template para extrair nomes de medicamentos e substâncias
    EXTRACT_MEDS = PromptTemplate(
        (
            "És um assistente especializado em identificar nomes de medicamentos e substâncias ativas.\n"
            "Extraia todos os nomes comerciais de medicamentos e/ou substâncias ativas mencionados no texto a seguir.\n"
            "Se for apenas mencionado os nomes comerciais de medicamentos, adiciona à resposta as substâncias ativas associadas aos medicamentos"
            "Retorne apenas os nomes separados por vírgula, sem pontos finais ou explicações adicionais.\n"
            "Se não forem mencionados nomes comerciais de medicamentos e/ou substâncias ativas devolve -\n\n"
            "Texto: {query}"
        )
    )
    
    # Template para fornecer informações adicionais sobre medicamentos
    MED_INFO = PromptTemplate(
        (
            "És um farmacêutico especializado que fornece informações sobre medicamentos.\n"
            "Para cada medicamento ou substância ativa mencionada abaixo, forneça informações detalhadas sobre:\n"
            "- Princípios ativos\n"
            "- Efeitos secundários\n"
            "- Classe do medicamento\n"
            "- Interações com outros medicamentos\n"
            "- Dosagem comum (se aplicável)\n"
            "- Contraindicações\n\n"
            "Medicamentos/Substâncias: {products}\n"
            "Contexto adicional da consulta: {query}\n\n"
            "Forneça informações completas, precisas e organizadas por medicamento."
        )
    )

    # Template para QA com contexto
    TEXT_QA = PromptTemplate(
        (
            "Contexto disponível abaixo.\n---------------------\n{context_str}\n---------------------\n"
            "Utilizando apenas o contexto fornecido e não conhecimento prévio, responda "
            "à pergunta: {query_str}\n"
            "Se o contexto não for útil, pode responder com o seu conhecimento próprio.\n"
            "Responda sempre em português de Portugal e não utilize palavras específicas do Brasil.\n"
            "Cite a fonte da informação se a utilizar.\n"
            "Resposta:\n"
        )
    )
    
    # Template para refinamento de respostas 
    REFINE = PromptTemplate(
        (
            "A pergunta original é: {query_str}\n"
            "Temos uma resposta existente: {existing_answer}\n"
            "Temos oportunidade de refinar a resposta existente (apenas se necessário) "
            "com mais contexto abaixo.\n------------\n{context_msg}\n------------\n"
            "Utilizando o contexto e não conhecimento prévio, atualize ou repita a resposta existente.\n"
            "Responda sempre em português de Portugal e não utilize palavras específicas do Brasil.\n"
            "Cite a fonte da nova informação se a utilizar.\n"
            "Se o contexto não for útil, mantenha a resposta original.\n"
            "Resposta Refinada: "
        )
    )
    
    # Template para reflexão sobre a qualidade da resposta
    REFLECTION = PromptTemplate(
        (
            "Analise criticamente a seguinte resposta para uma consulta sobre medicamentos:\n"
            "Pergunta: {query}\n"
            "Resposta: {response}\n\n"
            "Identifique problemas como:\n"
            "1. Informações incorretas ou imprecisas\n"
            "2. Falta de dados relevantes\n"
            "3. Inconsistências ou contradições\n"
            "4. Uso de português brasileiro em vez de português de Portugal\n"
            "5. Falta de citações de fontes\n\n"
            "Por favor, forneça uma avaliação e sugestões específicas para melhorar a resposta."
        )
    )
        
    # Template para integrar informações e criar uma resposta final
    INTEGRATE = PromptTemplate(
        (
            "És um consultor farmacêutico que integra informações de várias fontes para responder a consultas sobre medicamentos.\n"
            "Por favor, crie uma resposta completa e bem estruturada com base nas seguintes informações:\n\n"
            "Pergunta original: {query}\n\n"
            "Informações recuperadas da base de dados: {retrieved_info}\n\n"
            "Informações adicionais sobre os medicamentos: {med_info}\n\n"
            "Reflexões sobre possíveis problemas: {reflection}\n\n"
            "Responda de forma clara, precisa e completa em português de Portugal.\n"
            "Cite fontes quando apropriado e estruture sua resposta de forma lógica.\n"
            "Inclua informações sobre dosagem, efeitos secundários, interações e contraindicações quando relevantes."
        )
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
    
    def summarize(self) -> Dict[str, Any]:
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
        
    def create_base_filters(self, products: str) -> Dict[str, List[str]]:
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
            transformations=[text_splitter],
        )
    
    def create_query_engine(self, filters_qdrant):
        """Cria um motor de consulta com base nos filtros fornecidos"""
        retriever = VectorIndexRetriever(
            vector_store_kwargs={"qdrant_filters": filters_qdrant},
            index=self.index,
            similarity_top_k=30,
        )
        
        # Adicionar reranker para melhorar relevância
        #reranker = CohereRerank(api_key=COHERE_API_KEY, top_n=15)
        
        # Montar motor de consulta
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
        #    node_postprocessors=[reranker],
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
    
    def get_medication_info(self, products: str, query: str) -> str:
        """Obtém informações detalhadas sobre medicamentos"""
        return self.llm.predict(PromptTemplates.MED_INFO, products=products, query=query)
    
    def retrieve_information(self, query: str, products: str) -> Dict[str, Any]:
        """Recupera informações da base de conhecimento"""
        # Criar filtros para consulta
        filters_qdrant = self.filter_system.get_qdrant_filters(products)
        
        # Criar motor de consulta
        query_engine = self.index_retriever.create_query_engine(filters_qdrant)
        
        # Enriquecer consulta com contexto adicional
        med_info = self.get_medication_info(products, query)
        enhanced_query = f"{query}\n---------\nContexto e mais informações sobre os produtos:\n{med_info}"
        
        # Executar consulta
        result = query_engine.query(enhanced_query)
        
        # Extrair contextos
        contexts = [node.text for node in result.source_nodes]
        
        # Armazenar na memória
        self.memory.add_contexts(query, contexts)
        
        return {
            "response": result.response,
            "contexts": contexts,
            "enhanced_query": enhanced_query,
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
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Processa uma consulta completa através do sistema AgenticRAG"""
        start_time = time.time()
        
        # Adicionar consulta à memória
        self.memory.add_query(query)
        
        # 1. Extrair medicamentos mencionados
        products = self.extract_medications(query)
        print(f"Medicamentos detectados: {products}")
        
        # 2. Recuperar informações relevantes da base de conhecimento
        retrieval_result = self.retrieve_information(query, products)
        
        # 3. Avaliar a resposta inicial
        reflection = self.evaluate_response(query, retrieval_result["response"])
        
        # 4. Integrar todas as informações em uma resposta final
        final_response = self.integrate_information(
            query,
            retrieval_result["response"],
            retrieval_result["med_info"],
            reflection
        )
        
        end_time = time.time()
        execution_time = round(end_time - start_time)
        
        # Preparar resultado final
        result = {
            "query": query,
            "products_detected": products,
            "response": final_response,
            "contexts": retrieval_result["contexts"],
            "enhanced_query": retrieval_result["enhanced_query"],
            "execution_time": f"{execution_time}s"
        }
        
        return result


# Função principal para processar consultas
def answer_medication_query(query: str) -> Dict[str, Any]:
    """Função principal para processar consultas de medicamentos"""
    system = AgenticRagSystem()
    return system.process_query(query)


# Exemplo de uso
if __name__ == "__main__":
    # Exemplo de consulta
    query = "Quais cuidados devem ser tomados com pacientes acima de 65 anos ao usar RINVOQ?"
    
    print("Processando consulta...")
    result = answer_medication_query(query)
    
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