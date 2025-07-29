import timeit

import pandas as pd

from app import client
from app.hybrid import build_rag_pipeline as hrag
from app.initial_rag import build_rag_pipeline as irag
from app.support import generate_queries
from app.agentic import agent_process_query as arag

metadatasource = pd.read_csv("finaldbpt2.csv", delimiter=",")



def present_result(query):
    start = timeit.default_timer()

    products, add_info = generate_queries(query)

    print("Detected products:", products)
    rag_chain = irag(products=products, metadatasource=metadatasource)
    nquery = (
        query
        + "\n---------\nContext and more information about the products:\n"
        + add_info
    )

    current_app.logger.info("Pergunta melhorada: {}".format(nquery))

    answer = rag_chain.query(nquery)
    end = timeit.default_timer()

    return {
        "response": answer.response,
        "metadata": answer.metadata,
        "time": str(round(end - start)) + "s",
    }


def present_result_filtered(query, product, dosagem, method="hybrid"):
    start = timeit.default_timer()

    # _, add_info = generate_queries(query)

    # print("Detected products:", products)
    if method == "hybrid":
        rag_chain = hrag(
            client=client,
            products=product,
            metadatasource=metadatasource,
            strength=dosagem,
        )
    elif method == "agentic":
        rag_chain = arag(
            query=query,
            client=client,
            metadatasource=metadatasource,
            products=product,
            strength=dosagem) 

    elif method == "initial":
        rag_chain = irag(
            client=client,
            products=product,
            metadatasource=metadatasource,
            strength=dosagem,
        )
    else:
        return "Error unknown method"
    # nquery = (
    #     query
    #     + "\n---------\nContext and more information about the products:\n"
    #     + add_info
    # )
    afterrag = timeit.default_timer()
    print("rag_chain took " + str(round(afterrag - start)))
    # app.logger.info("Pergunta melhorada: {}".format(query))

    answer = rag_chain.query(query)
    afterrag2 = timeit.default_timer()

    print("rag_chain.query took " + str(round(afterrag2 - afterrag)))

    end = timeit.default_timer()
    # print(answer)
    return {
        "response": answer.response,
        "metadata": answer.metadata,
        "contexts": answer.source_nodes,
        "full": answer,
        "time": str(round(end - start)) + "s",
    }
