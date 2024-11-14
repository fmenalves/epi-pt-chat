import os

import pandas as pd
import qdrant_client
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFium2Loader
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

os.environ["OPENAI_API_KEY"] = "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"
index_name = "Pardal"

client = qdrant_client.QdrantClient(
    "http://localhost:6333",
    # api_key="<qdrant-api-key>", # For Qdrant Cloud, None for local instance
)


embed_model_name = "sentence-transformers/all-mpnet-base-v2"

metadatasource = pd.read_csv("finaldbpt-dev.csv", delimiter=",")


def get_metadata(metadatasource, nfilename, file_path):
    extra_metadata = metadatasource[metadatasource["NewFilename"] == nfilename].values

    if len(extra_metadata) == 0:
        nfilename = file_path.split("/")[-1][:-4]
        extra_metadata = metadatasource[
            metadatasource["NewFilename"] == nfilename
        ].values

    if len(extra_metadata) == 0:
        print("No metadata found for", nfilename)
        extra_metadata = [["", "", "", "", "", ""]]
    ff = {
        "Nome_Comercial": extra_metadata[0][1],
        "substancia": extra_metadata[0][2],
        "Forma_Farmaceutica": extra_metadata[0][3],
        "Dosagem": extra_metadata[0][4],
        "Titular_AIM": extra_metadata[0][5],
    }

    return ff


def build_index_big(file_list, client, index_name):
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=embed_model_name))
    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=64)

    Settings.text_splitter = text_splitter
    Settings.embed_model = embed_model
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 64

    vector_store = QdrantVectorStore(client=client, collection_name=index_name)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Iterate over each file
    for idx, file_path in enumerate(file_list):
        # if idx!=0 and idx%100==0:
        #     break
        # print(file_path)
        nfilename = file_path.split("/")[-1]

        loader = PyPDFium2Loader(file_path)

        ff = get_metadata(metadatasource, nfilename, file_path)
        # Load and parse the file contents
        parsed_data = loader.load()
        for p in parsed_data:
            p.metadata.update(ff)

        if idx % 500 == 0:
            print("Loading documents...", idx)
        documents = [
            Document(text=t.page_content, metadata=t.metadata) for t in parsed_data
        ]
        try:
            index = VectorStoreIndex.from_documents(
                documents,
                # transformations=[text_splitter],
                storage_context=storage_context,
            )
        except Exception as e:
            print("Error in", nfilename, e)

    return True


file_list = [
    "/Users/joaoalmeida/Desktop/PDH/epi-gather/bulas-final/Comirnaty_Original_Omicron_BA_4_5Dispersão_injetável30_µg_15_µg_+_15_µg_0_3_ml.pdf",
    "/Users/joaoalmeida/Desktop/PDH/epi-gather/bulas-final/Comirnaty_Omicron_XBB_1_5Dispersão_injetável10_µg__0_3_ml.pdf",
    "/Users/joaoalmeida/Desktop/PDH/epi-gather/bulas-final/Comirnaty_Omicron_XBB_1_5Concentrado_para_dispersão_injetável3_µg__0_2_ml.pdf",
    "/Users/joaoalmeida/Desktop/PDH/epi-gather/bulas-final/Comirnaty_Omicron_XBB_1_5Concentrado_para_dispersão_injetável10_µg_0_2_ml.pdf",
    "/Users/joaoalmeida/Desktop/PDH/epi-gather/bulas-final/ComirnatyDispersão_injetável30_mcg_0_3_ml.pdf",
    "/Users/joaoalmeida/Desktop/PDH/epi-gather/bulas-final/Comirnaty_KP_2Dispersão_injetável_em_seringa_pré_cheia30_µg__0_3_ml.pdf",
    "/Users/joaoalmeida/Desktop/PDH/epi-gather/bulas-final/ComirnatyConcentrado_para_dispersão_injetável3_mcg_0_2_ml.pdf",
    "/Users/joaoalmeida/Desktop/PDH/epi-gather/bulas-final/ComirnatyConcentrado_para_dispersão_injetável10_mcg_0_2_ml.pdf",
]

file_list = [
    "/Users/joaoalmeida/Desktop/PDH/epi-gather/bulas-final/Comirnaty_JN_1Dispersão_injetável_em_seringa_pré_cheia30_µg_0_3_ml.pdf"
]

build_index_big(file_list, client, index_name)
