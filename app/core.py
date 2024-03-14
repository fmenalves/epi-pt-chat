from llama_index.core import VectorStoreIndex

# from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
import timeit
import time
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SentenceSplitter

from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from dotenv import load_dotenv
import os

load_dotenv()


INDEX_NAME = os.getenv("INDEX_NAME")
URI_BD = os.getenv("URI_BD")
LLM_URL = os.getenv("LLM_URL")
# client = OpenAI(
#    # This is the default and can be omitted
#    api_key=os.getenv("OPENAI_KEY"),
# )


def retrieve_index(chunk_size, llm, index_name):
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
    Settings.text_splitter = text_splitter

    # embed_model = OpenAIEmbedding(embed_batch_size=10)
    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
    )
    Settings.llm = llm
    Settings.embed_model = embed_model
    # Settings.num_output = 512
    # Settings.context_window = 3900
    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = 64

    vector_store = LanceDBVectorStore(uri=URI_BD, table_name=index_name)

    index = VectorStoreIndex.from_vector_store(
        vector_store,  # service_context=service_context
        transformations=[text_splitter],
    )

    return index


def build_rag_pipeline(debug=True):
    llm = Ollama(model="mistral", base_url=LLM_URL, temperature=0)
    # llm=OpenAI(api_key="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",model="gpt-4")
    print("Building index...")
    index = retrieve_index(20000, llm, index_name=INDEX_NAME)
    print("Constructing query engine...")
    retriever = VectorIndexRetriever(
        index=index,
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
        #     reranker=reranker,  # Include the reranker here
        #  response_synthesizer=response_synthesizer,
        node_postprocessors=[
            reranker
        ],  # ,SimilarityPostprocessor(similarity_cutoff=0.7)],
    )
    return query_engine


def present_result(query):
    start = timeit.default_timer()

    rag_chain = build_rag_pipeline(debug=True)

    step = 0
    answer = False
    while not answer:
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
