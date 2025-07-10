# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
import os

from flask import current_app
from llama_index.core import Settings, VectorStoreIndex, get_response_synthesizer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore

from app import app
from app.support import (
    get_filters_qdrant,
    get_filters_qdrant_filtered,
    text_qa_template,
)

index_name = os.getenv("INITIAL_INDEX_NAME")

embed_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embed_model_name = "sentence-transformers/all-mpnet-base-v2"


def retrieve_index(client, llm, index_name):
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
    Settings.text_splitter = text_splitter

    # embed_model = OpenAIEmbedding(embed_batch_size=10)
    # embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=embed_model_name))

    ollama_embedding = OllamaEmbedding(
        model_name="nomic-embed-text",
        # base_url="http://localhost:11434",
        # ollama_additional_kwargs={"mirostat": 0},
    )
    embed_model = ollama_embedding
    Settings.llm = llm
    Settings.embed_model = embed_model
    # Settings.num_output = 512
    # Settings.context_window = 3900
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 64

    vector_store = QdrantVectorStore(client=client, collection_name=index_name)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        text_qa_template=text_qa_template,
        # refine_template=refine_template,
        transformations=[text_splitter],
    )

    return index


def build_rag_pipeline(client, products, metadatasource, strength=None):
    if current_app.config["OPENAI_KEY"] is not None:
        llm = OpenAI(
            temperature=0, api_key=current_app.config["OPENAI_KEY"], model="gpt-4"
        )
    else:
        # pass
        llm = Ollama(
            # model="llama3.1:70b",
            model="llama3.3",
            base_url=current_app.config["LLM_URL"],
            temperature=0,
            request_timeout=60,
        )
    print("Building index...")
    index = retrieve_index(client, llm, index_name)
    print("Constructing query engine...")
    if strength:  # demo
        filters_qdrant = get_filters_qdrant_filtered(
            products=products, metadatasource=metadatasource, strength=strength
        )
    else:  # not demo
        filters_qdrant = get_filters_qdrant(
            products=products, metadatasource=metadatasource
        )
    print("filtro", filters_qdrant)
    app.logger.info("Filtros: {}".format(filters_qdrant))

    results = client.search(
        collection_name=index_name,
        query_vector=[0.1] * 768,
        limit=1,
        query_filter=filters_qdrant,
    )

    # Check if results are found
    if len(results) == 0:
        print("issue with the product and data in the collection")
        filters_qdrant = None
    retriever = VectorIndexRetriever(
        vector_store_kwargs={"qdrant_filters": filters_qdrant},
        index=index,
        # filters=filters,
        similarity_top_k=2,
    )
    # configure response synthesizer
    # reranker = CohereRerank(api_key=cohere_api_key, top_n=2)
    response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT)
    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        #    reranker
        #  ],  # ,SimilarityPostprocessor(similarity_cutoff=0.7)],
    )

    query_engine.update_prompts(
        {
            "response_synthesizer:text_qa_template": text_qa_template,
            #   "response_synthesizer:refine_template": refine_template,
        }
    )

    return query_engine
