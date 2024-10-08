from llama_index.core.vector_stores.types import (
    MetadataFilters,
    MetadataFilter,
    ExactMatchFilter,
)
from llama_index.core import PromptTemplate
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import os
from llama_index.llms.ollama import Ollama

load_dotenv()


LLM_URL = os.getenv("LLM_URL")


OPENAI_KEY = os.getenv("OPENAI_KEY")

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


def filter_lamma(metadatasource, products):
    filters = []
    f = create_filters(products=products, metadatasource=metadatasource)
    if len(f["Nome_Comercial"]) > 0:
        filters.append(
            ExactMatchFilter(key="Nome_Comercial", value=list(set(f["Nome_Comercial"])))
        )
    if len(f["Substancia"]) > 0:
        filters.append(
            ExactMatchFilter(key="Substancia", value=list(set(f["Substancia"])))
        )
    # return f
    if len(filters) == 0:
        return MetadataFilters(filters=None)
    if len(filters) == 1:
        return MetadataFilters(filters=filters)
    return MetadataFilters(filters=filters, condition="or")


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


def generate_queries(query: str):
    query_gen_str = """You are a helpful assistant that can provide information about drugs. Please comercial drug names and/or substances, separated by a comma. You will return only the names separeted by comma and nothing more. The text to analyse is {query}."""
    query_gen_prompt1 = PromptTemplate(query_gen_str)
    query_gen_str2 = """You are a helpful assistant that can provide information about drugs. Please provide information additional about the drugs detected in the query. Important things to mention are active principles, side effects, drug class and interactions with other drugs for each detected drug.
Query: {query}
"""
    query_gen_prompt2 = PromptTemplate(query_gen_str2)
    if OPENAI_KEY is not None:
        llm = OpenAI(api_key=OPENAI_KEY, model="gpt-3.5-turbo")
    else:
        # pass
        llm = Ollama(
            model="llama3.1:70b",
            base_url=LLM_URL,
            temperature=0,
            request_timeout=120,
        )

    products = llm.predict(query_gen_prompt1, query=query)
    # assume LLM proper put each query on a newline

    add_info = llm.predict(query_gen_prompt2, query=query)
    return products.replace(".", ""), add_info
