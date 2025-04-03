import os
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from dotenv import load_dotenv
import pandas as pd

from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.embeddings import HuggingFaceEmbeddings #from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
import qdrant_client

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.query_engine import RetrieverQueryEngine



#docker pull qdrant/qdrant
#docker run -p 6333:6333 -p 6334:6334 -v %cd%/qdrant_storage:/qdrant/storage:z -d qdrant/qdrant

load_dotenv()

LLM_URL = os.getenv("LLM_URL")
URI_BD = os.getenv("URI_BD")
cohere_api_key = os.getenv("COHERE_API_KEY")
index_name = os.getenv("INDEX_NAME")

client = qdrant_client.QdrantClient(URI_BD)

text_qa_template_str = (
    "Context information is"
    " below.\n---------------------\n{context_str}\n---------------------\nUsing"
    " the context information and not prior knowledge, answer"
    " the question: {query_str}\nIf the context isn't helpful, you can also"
    " answer the question on your own."
    "Respond always in the language portuguese from portugal and do not use any word that is specific from brazil."
    "Cite the source of the new information if you use it.\n"
    "Answer:\n"
)
text_qa_template = PromptTemplate(text_qa_template_str)

refine_template_str = (
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

refine_template = PromptTemplate(refine_template_str)


#if OPENAI_KEY is not None:
#    llm = OpenAI(api_key=OPENAI_KEY, model="gpt-3.5-turbo")
#else:
    ## pass
llm = Ollama(
    model="llama3.1:70b",
    base_url=LLM_URL,
    temperature=0,
    request_timeout=120,
)


#embed_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embed_model_name = "sentence-transformers/all-mpnet-base-v2"

metadatasource = pd.read_csv("finaldbpt2.csv", delimiter=",")



def generate_queries(query: str):
    query_gen_str = """You are a helpful assistant that can provide information about drugs. Please comercial drug names and/or substances, separated by a comma. You will return only the names separeted by comma and nothing more. The text to analyse is {query}."""
    query_gen_prompt1 = PromptTemplate(query_gen_str)
    query_gen_str2 = """You are a helpful assistant that can provide information about drugs. Please provide information additional about the drugs detected in the query. Important things to mention are active principles, side effects, drug class and interactions with other drugs for each detected drug.
    Query: {query}
    """
    query_gen_prompt2 = PromptTemplate(query_gen_str2)
#if OPENAI_KEY is not None:
#    llm = OpenAI(api_key=OPENAI_KEY, model="gpt-3.5-turbo")
#else:
    ## pass

    products = llm.predict(query_gen_prompt1, query=query)
    # assume LLM proper put each query on a newline

    add_info = llm.predict(query_gen_prompt2, query=query)

    return products.replace(".", ""), add_info





def create_filters(products, metadatasource):

    f = {"Nome_Comercial": [], "Substancia": []}
    for word in products.split(","):
        # print(word)
        word = word.strip()
        if any(
            word.lower() == item.lower() for item in metadatasource["Nome Comercial"]
        ):
            # print("found")
            val = metadatasource[
                [
                    word.lower() == item.lower()
                    for item in metadatasource["Nome Comercial"]
                ]
            ]["Nome Comercial"].values
            print(word, "comer")
            for v in val:
                f["Nome_Comercial"].append(v)
        #  f.append(MetadataFilter(key="Nome_Comercial", value=val, operator="=="))
        elif any(word.lower() == str(item).lower() for item in metadatasource["Subs"]):
            val = metadatasource[
                [word.lower() == str(item).lower() for item in metadatasource["Subs"]]
            ]["Subs"].values

            for v in val:
                # f[v] = "Substancia"
                f["Substancia"].append(v)

        if metadatasource["Nome Comercial"].str.contains(word, case=False).any():
            # print("found")
            val = metadatasource[
                metadatasource["Nome Comercial"].str.contains(word, case=False)
            ]["Nome Comercial"].values
            print(word, "comer")
            for v in val:
                # f[v] = "Nome_Comercial"
                f["Nome_Comercial"].append(v)

        #  f.append(MetadataFilter(key="Nome_Comercial", value=val, operator="=="))
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
                # f[v] = "Substancia"
                f["Substancia"].append(v)

    return f








def get_filters_qdrant(metadatasource, products):
    filters = []
    f = create_filters(products=products, metadatasource=metadatasource)

    if len(f["Nome_Comercial"]) > 0:
        filters.append(
            FieldCondition(
                key="Nome_Comercial",
                match=MatchAny(any=list(set(f["Nome_Comercial"]))),
            )
        )
    if len(f["Substancia"]) > 0:
        filters.append(
            FieldCondition(
                key="Substancia",
                match=MatchAny(any=list(set(f["Nome_Comercial"]))),
            )
        )
        # return f
    if len(filters) == 0:
        return Filter(should=[])
    if len(filters) > 0:
        return Filter(should=filters)




def retrieve_index(client, llm, index_name):
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
    Settings.text_splitter = text_splitter

    ## embed_model = OpenAIEmbedding(embed_batch_size=10)
    embed_model = HuggingFaceEmbeddings(model_name=embed_model_name)
    Settings.llm = llm
    Settings.embed_model = embed_model
    ## Settings.num_output = 512
    ## Settings.context_window = 3900
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





def build_rag_pipeline(products, metadatasource):
        
    print("Building index...")
    index = retrieve_index(client, llm, index_name)
    print("Constructing query engine...")
    filters_qdrant = get_filters_qdrant(
            products=products, metadatasource=metadatasource
        )
    print(filters_qdrant)
    #    app.logger.info("Filtros: {}".format(print(filters_qdrant)))

    retriever = VectorIndexRetriever(
        vector_store_kwargs={"qdrant_filters": filters_qdrant},
        index=index,
        # filters=filters,
        similarity_top_k=30,
    )
    # configure response synthesizer
    # response_synthesizer = get_response_synthesizer()
    # reranker = SentenceTransformerRerank(
    #    model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=15
    # )

    cohere_api_key = os.getenv("COHERE_API_KEY")

    reranker = CohereRerank(api_key=cohere_api_key, top_n=15)
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





def present_result_melhorado(query):
#    start = timeit.default_timer()

    products, add_info = generate_queries(query)

    print("Detected products:", products)
    rag_chain = build_rag_pipeline(products=products, metadatasource=metadatasource)
    nquery = (
        query
        + "\n---------\nContext and more information about the products:\n"
        + add_info
    )

    print("Pergunta melhorada: {}".format(nquery))

#    app.logger.info("Pergunta melhorada: {}".format(nquery))

    answer = rag_chain.query(nquery)
#    end = timeit.default_timer()

    node_context = answer.source_nodes

    contexts = []

    for i in range(len(node_context)):


        contexts.append(node_context[i].text)

    return {
        "response": answer.response,
        "context": contexts,
        "nquery" : nquery
#        "time": str(round(end - start)) + "s",
    }

def present_result(query):
#    start = timeit.default_timer()

    products, add_info = generate_queries(query)

    print("Detected products:", products)
    rag_chain = build_rag_pipeline(products=products, metadatasource=metadatasource)
#    nquery = (
#        query
#        + "\n---------\nContext and more information about the products:\n"
#        + add_info
#    )

    #print("Pergunta melhorada: {}".format(nquery))

#    app.logger.info("Pergunta melhorada: {}".format(nquery))

    answer = rag_chain.query(query)
#    end = timeit.default_timer()

    node_context = answer.source_nodes

    contexts = []

    for i in range(len(node_context)):


        contexts.append(node_context[i].text)

    return {
        "response": answer.response,
        "context": contexts
#        "time": str(round(end - start)) + "s",
    }


