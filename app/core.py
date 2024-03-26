from llama_index.core import VectorStoreIndex
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
import timeit
import time
from llama_index.vector_stores.qdrant import QdrantVectorStore

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SentenceSplitter

from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from dotenv import load_dotenv
import os
from app.support import text_qa_template, refine_template, create_filters
import qdrant_client
import pandas as pd

load_dotenv()


index_name = os.getenv("INDEX_NAME")
URI_BD = os.getenv("URI_BD")
LLM_URL = os.getenv("LLM_URL")
# client = OpenAI(
#    # This is the default and can be omitted
#    api_key=os.getenv("OPENAI_KEY"),
# )
metadatasource = pd.read_csv("finaldbpt2.csv", delimiter=",")
embed_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embed_model_name = "sentence-transformers/all-mpnet-base-v2"
client = qdrant_client.QdrantClient(URI_BD)


def retrieve_index(client, llm, index_name):
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
    Settings.text_splitter = text_splitter

    # embed_model = OpenAIEmbedding(embed_batch_size=10)
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=embed_model_name))
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
        refine_template=refine_template,
        transformations=[text_splitter],
    )

    return index


def build_rag_pipeline(query, metadatasource):
    llm = Ollama(
        model="mistral",
        base_url=LLM_URL,
        temperature=0,
        request_timeout=120,
    )
    # llm=OpenAI(api_key="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",model="gpt-4")
    print("Building index...")
    index = retrieve_index(client, llm, index_name)
    print("Constructing query engine...")
    filters = create_filters(query, metadatasource)
    print(filters)
    retriever = VectorIndexRetriever(
        index=index,
        filters=filters,
        similarity_top_k=20,
    )
    # configure response synthesizer
    # response_synthesizer = get_response_synthesizer()
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=10
    )
    #   reranker = CohereRerank(top_n=10)
    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,  # response_mode="compact",
        node_postprocessors=[
            reranker
        ],  # ,SimilarityPostprocessor(similarity_cutoff=0.7)],
    )

    query_engine.update_prompts(
        {
            "response_synthesizer:text_qa_template": text_qa_template,
            "response_synthesizer:refine_template": refine_template,
        }
    )

    return query_engine


def present_result(query):
    start = timeit.default_timer()

    rag_chain = build_rag_pipeline(query, metadatasource)

    step = 0
    answer = False
    while not answer:
        print("prompts", rag_chain.get_prompts())

        step += 1
        if step > 1:
            print("Refining answer...")
            # add wait time, before refining to avoid spamming the server
            time.sleep(5)
        if step > 3:
            # if we have refined 3 times, and still no answer, break
            answer = "No answer found."
            break
        print("Retrieving answer...")

        # answer = get_rag_response(query, rag_chain, debug=True)
        answer = rag_chain.query(query)
    #  print(answer.response)
    end = timeit.default_timer()

    print(answer)

    return {
        "response": answer.response,
        "metadata": answer.metadata,
        "time": str(round(end - start)) + "s",
    }
