import json
import os

import numpy as np
import pandas as pd
import qdrant_client
from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding
from langchain_community.document_loaders import PyPDFium2Loader
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

# ==== CONFIG ====
index_name = "full-hybrid-search"
embed_model_name = "nomic-embed-text"
pdf_folder = "bulas-final"
csv_path = "finaldbpt2.csv"

# ==== CLIENT SETUP ====
client = qdrant_client.QdrantClient("http://localhost:6333", timeout=600)
metadatasource = pd.read_csv(csv_path, delimiter=",")
print(metadatasource.head(4))
# ==== TF-IDF SETUP (BM25-like sparse vector generation) ====
# vectorizer = TfidfVectorizer()
dense_embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25")
late_interaction_embedding_model = LateInteractionTextEmbedding(
    "colbert-ir/colbertv2.0"
)


def get_metadata(metadatasource, nfilename, file_path):
    base = nfilename.replace(".pdf", "")
    print(base)
    row = metadatasource[metadatasource["filenameclean"] == base]
    if row.empty:
        print(f"No metadata found for {base}")
        return {}

    r = row.iloc[0]
    return {
        "Nome_Comercial": r["Nome Comercial"],
        "substancia": r["Subs"],
        "Forma_Farmaceutica": r["FF"],
        "Dosagem": r["Dosage"],
        "Titular_AIM": r["TAIM"],
    }


import time

from qdrant_client.http.exceptions import UnexpectedResponse


def chunked_upsert(client, collection_name, points, chunk_size=100):
    total = len(points)
    for i in range(0, total, chunk_size):
        chunk = points[i : i + chunk_size]
        try:
            print(f"Uploading chunk {i} to {min(i+chunk_size, total)}...")
            client.upsert(collection_name=collection_name, points=chunk)
        except UnexpectedResponse as e:
            print(f"❌ Chunk {i}-{i+chunk_size} failed:", e)
        time.sleep(0.1)  # optional pause to avoid overwhelming the server


def build_index(file_list, client, index_name):
    # Embeddings (dense)
    embed_model = OllamaEmbedding(model_name=embed_model_name)
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    Settings.embed_model = embed_model
    Settings.text_splitter = splitter

    # Build TF-IDF vocabulary from all documents (single pass)
    all_texts = []
    parsed_docs = []

    for file_path in file_list:
        loader = PyPDFium2Loader(file_path)
        filename = os.path.basename(file_path)
        metadata = get_metadata(metadatasource, filename, file_path)
        pages = loader.load()
        for p in pages:
            p.metadata.update(metadata)
            all_texts.append(p.page_content)
            parsed_docs.append(p)

    print("Fitting TF-IDF vocabulary...")
    # vectorizer.fit(all_texts)

    # Generate and upsert each point
    print("Indexing...")
    # for i, p in enumerate(parsed_docs):
    # dense_vector = embed_model.get_text_embedding(p.page_content)
    # sparse_vector = generate_sparse_vector(p.page_content)
    dense_embeddings = list(
        dense_embedding_model.embed(p.page_content for p in parsed_docs)
    )
    bm25_embeddings = list(
        bm25_embedding_model.embed(p.page_content for p in parsed_docs)
    )
    late_interaction_embeddings = list(
        late_interaction_embedding_model.embed(p.page_content for p in parsed_docs)
    )
    points = []
    client.recreate_collection(  # usar recreate para testes, apaga se já existir
        collection_name=index_name,
        vectors_config={
            "all-MiniLM-L6-v2": VectorParams(size=384, distance=Distance.COSINE),
            "colbertv2.0": VectorParams(size=128, distance=Distance.COSINE),
        },
        sparse_vectors_config={"bm25": SparseVectorParams(modifier="idf")},
    )
    for idx, (
        dense_embedding,
        bm25_embedding,
        late_interaction_embedding,
        doc,
    ) in enumerate(
        zip(dense_embeddings, bm25_embeddings, late_interaction_embeddings, parsed_docs)
    ):
        metadata = doc.metadata.copy()
        metadata["page"] = metadata.get("page", idx)
        payload = {
            "text": doc.page_content,
            "metadata": metadata,
        }
        bm25_obj = bm25_embedding.as_object()
        print(len(late_interaction_embedding), len(late_interaction_embedding[0]))
        # assert len(dense_embedding) == 384
        # assert len(late_interaction_embedding) == 128

        # Average all 512 vectors → 1 vector of shape (128,)
        late_embedding = np.mean(late_interaction_embedding, axis=0).tolist()
        point = PointStruct(
            id=idx,
            vector={
                "all-MiniLM-L6-v2": dense_embedding,  # list of floats
                "colbertv2.0": late_embedding,  # list of floats
                "bm25": bm25_obj,
            },
            payload=payload,
        )
        print(json.dumps(point.model_dump(), indent=2))
        # If you're using Pydantic v2:
        point_dict = point.model_dump()

        # (if using Pydantic v1, use point.dict() instead)

        # Save to file
        with open("point_sample.json", "w", encoding="utf-8") as f:
            json.dump(point_dict, f, indent=2, ensure_ascii=False)
        points.append(point)
        client.upsert(collection_name=index_name, points=[point])

    # print(points)
    #  print(len(points))
    #  print(len(points[0]))
    # operation_info = client.upsert(collection_name=index_name, points=points,  )
    # chunked_upsert(client, index_name, points, chunk_size=10)  # tune size as needed
    print("Done.")


# ==== Collect PDF paths ====
file_list = [
    os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")
]
file_list = [
    "bulas-final/DiovanComprimido_revestido_por_película80_mg.pdf",
    "bulas-final/Lenalidomida_Tecnigen_MGCápsula10_mg.pdf",
    "bulas-final/Ciplox_MGComprimido_revestido_por_película500_mg.pdf",
    "bulas-final/RinvoqComprimido_de_libertação_prolongada15_mg.pdf",
    "bulas-final/OzempicSolução_injetável_em_caneta_pré_cheia0_5_mg_0_37_ml.pdf",
    "bulas-final/Paclitaxel_Accord_MGConcentrado_para_solução_para_perfusão6_mg_ml.pdf",
    "bulas-final/Influvac_TetraSuspensão_injetável_em_seringa_pré_cheiaAssociação.pdf",
    "bulas-final/Comirnaty_JN_1Dispersão_injetável_em_seringa_pré_cheia30_µg_0_3_ml.pdf",
    "bulas-final/TriticumComprimido_revestido_por_película100_mg.pdf",
    "bulas-final/DepakineXarope40_mg_ml.pdf",
]


# ==== Run Indexing ====
build_index(file_list, client, index_name)
