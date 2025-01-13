import os

import pandas as pd
import qdrant_client
from langchain_community.document_loaders import PyPDFium2Loader
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

os.environ["OPENAI_API_KEY"] = "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"
index_name = "Coelho"  # nomic
DATA_TRANSFORM = {
    "Ciplox 500mg": ["Ciplox MG", "500 mg"],
    "RInvoq 15mg": ["Rinvoq", "15 mg"],
    "Ozempic 0.5mg": ["Ozempic", "0.5 mg/0.37 ml"],
    "Paclitaxel Accord 6mg": ["Paclitaxel Accord MG", "6 mg/ml"],
    "Diovan 80 mg": ["Diovan", "80 mg"],
    "Influvac tetra": ["Influvac Tetra", "Associação"],
    "Comirnaty JN.1": ["Comirnaty JN.1", "30 µg/0.3 ml"],
    "Triticum 100 mg": ["Triticum", "100 mg"],
    "Depakine 40 mg /ml": ["Depakine", "40 mg/ml"],
    "Lenalidomida tecnigen 10mg": ["Lenalidomida Tecnigen MG", "10 mg"],
}
client = qdrant_client.QdrantClient(
    "http://localhost:6333",
    # api_key="<qdrant-api-key>", # For Qdrant Cloud, None for local instance
)


embed_model_name = "nomic-embed-text"
metadatasource = pd.read_csv("../finaldbpt2.csv", delimiter=",")


def get_metadata(metadatasource, nfilename, file_path):
    extra_metadata = metadatasource[metadatasource["NewFilename"] == nfilename].values

    if len(extra_metadata) == 0:
        nfilename = file_path.split("/")[-1][:-4]
        extra_metadata = metadatasource[
            metadatasource["NewFilename"] == nfilename
        ].values

    if len(extra_metadata) == 0:
        print("No metadata found for", nfilename)
        raise Exception("No metadata found")
        # extra_metadata = [["", "", "", "", "", ""]]
    ff = {
        "Nome_Comercial": extra_metadata[0][1],
        "substancia": extra_metadata[0][2],
        "Forma_Farmaceutica": extra_metadata[0][3],
        "Dosagem": extra_metadata[0][4],
        "Titular_AIM": extra_metadata[0][5],
    }

    return ff


def build_index_big(file_list, client, index_name):
    # embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=embed_model_name))
    ollama_embedding = OllamaEmbedding(
        model_name="nomic-embed-text",
        # base_url="http://localhost:11434",
        # ollama_additional_kwargs={"mirostat": 0},
    )
    embed_model = ollama_embedding

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
        print(ff)
        # Load and parse the file contents
        parsed_data = loader.load()
        for p in parsed_data:
            p.metadata.update(ff)

        # if idx % 500 == 0:
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
    "/Users/joaoalmeida/Desktop/PDH/epi-gather/bulas-final/DiovanComprimido_revestido_por_película80_mg.pdf",
    "/Users/joaoalmeida/Desktop/PDH/epi-gather/bulas-final/Lenalidomida_Tecnigen_MGCápsula10_mg.pdf",
    "/Users/joaoalmeida/Desktop/PDH/epi-gather/bulas-final/Ciplox_MGComprimido_revestido_por_película500_mg.pdf",
    "/Users/joaoalmeida/Desktop/PDH/epi-gather/bulas-final/RinvoqComprimido_de_libertação_prolongada15_mg.pdf",
    "/Users/joaoalmeida/Desktop/PDH/epi-gather/bulas-final/OzempicSolução_injetável_em_caneta_pré_cheia0_5_mg_0_37_ml.pdf",
    "/Users/joaoalmeida/Desktop/PDH/epi-gather/bulas-final/Paclitaxel_Accord_MGConcentrado_para_solução_para_perfusão6_mg_ml.pdf",
    "/Users/joaoalmeida/Desktop/PDH/epi-gather/bulas-final/Influvac_TetraSuspensão_injetável_em_seringa_pré_cheiaAssociação.pdf",
    "/Users/joaoalmeida/Desktop/PDH/epi-gather/bulas-final/Comirnaty_JN_1Dispersão_injetável_em_seringa_pré_cheia30_µg_0_3_ml.pdf",
    "/Users/joaoalmeida/Desktop/PDH/epi-gather/bulas-final/TriticumComprimido_revestido_por_película100_mg.pdf",
    "/Users/joaoalmeida/Desktop/PDH/epi-gather/bulas-final/DepakineXarope40_mg_ml.pdf",
]

build_index_big(file_list, client, index_name)
