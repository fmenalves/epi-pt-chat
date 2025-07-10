import os
from typing import List, Optional

from fastembed import SparseTextEmbedding, TextEmbedding
from flask import current_app

# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
# import QueryBundle
from llama_index.core import (
    Settings,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.query_engine import CitationQueryEngine

# Retrievers
from llama_index.core.retrievers import (
    BaseRetriever,
)
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

# import NodeWithScore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.models import NamedSparseVector, NamedVector

from app import app
from app.support import (
    get_filters_qdrant,
    get_filters_qdrant_filtered,
    text_qa_template,
)

embed_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embed_model_name = "sentence-transformers/all-mpnet-base-v2"
index_name = os.getenv("HYBRID_INDEX_NAME")
MODEL = "llama3.2"


class CorrigirPageLabelPostprocessor(BaseNodePostprocessor):
    def _postprocess_nodes(self, nodes, query_str=None):
        for node in nodes:
            if "page" in node.metadata and "page_label" not in node.metadata:
                node.metadata["page_label"] = node.metadata["page"]
        return nodes


class CustomHybridRetriever(BaseRetriever):
    def __init__(
        self,
        client,
        index,
        dense_model,
        sparse_model,
        collection_name: str,
        qdrant_filters: Optional[dict] = None,
        alpha: float = 0.7,
        dense_vector_name: str = "all-MiniLM-L6-v2",
        sparse_vector_name: str = "bm25",
        top_k: int = 10,
    ):
        self.client = client
        self.index = index
        self.dense_model = dense_model
        self.sparse_model = sparse_model
        self.collection_name = collection_name
        self.filters = qdrant_filters
        self.alpha = alpha
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name
        self.top_k = top_k

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query = query_bundle.query_str

        # Gerar embeddings com fastembed
        dense_vector = list(self.dense_model.embed([query]))[0]
        sparse_vector = list(self.sparse_model.embed([query]))[0].as_object()

        # Buscar resultados de cada um
        dense_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=NamedVector(name=self.dense_vector_name, vector=dense_vector),
            query_filter=self.filters,
            limit=self.top_k,
            with_payload=True,
        )

        sparse_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=NamedSparseVector(
                name=self.sparse_vector_name, vector=sparse_vector
            ),
            query_filter=self.filters,
            limit=self.top_k,
            with_payload=True,
        )

        # Combinar scores
        combined_scores = {}

        def add_score(result, weight):
            _id = str(result.id)
            score = result.score * weight
            if _id in combined_scores:
                combined_scores[_id]["score"] += score
            else:
                combined_scores[_id] = {
                    "score": score,
                    "payload": result.payload,
                }

        for res in dense_results:
            add_score(res, self.alpha)
        for res in sparse_results:
            add_score(res, 1 - self.alpha)

        # Criar nodes
        results = []
        for _id, data in combined_scores.items():
            text = data["payload"].get("text", "")
            metadata = data["payload"].get("metadata", {})

            if not text:
                continue
            node = TextNode(text=text, metadata=metadata, id_=_id)
            results.append(NodeWithScore(node=node, score=data["score"]))

        return sorted(results, key=lambda x: x.score, reverse=True)[: self.top_k]


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
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    # vector_store = QdrantVectorStore(client=client, collection_name=index_name)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=index_name,
        enable_hybrid=True,  # ✅ ativa suporte a híbrido
        fastembed_sparse_model="Qdrant/bm25",  # modelo usado para sparse
    )
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
            model=MODEL,
            # model="gemma2",
            base_url=current_app.config["LLM_URL"],
            temperature=0,
            request_timeout=600,
            context_window=4096,  # Adjust the context size as needed
        )
    # llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)

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
        #   query_vector=[0.1] * 768,
        limit=1,
        # vector_name="all-MiniLM-L6-v2",  # ✅ IMPORTANTE
        query_vector=("all-MiniLM-L6-v2", [0.1] * 384),  # ✅ nome do vetor e vetor
        query_filter=filters_qdrant,
    )

    # Check if results are found
    if len(results) == 0:
        print("issue with the product and data in the collection")
        filters_qdrant = None

    dense_embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    sparse_embedding_model = SparseTextEmbedding("Qdrant/bm25")
    retriever = CustomHybridRetriever(
        client=client,
        index=index,
        dense_model=dense_embedding_model,
        sparse_model=sparse_embedding_model,
        collection_name="hybrid-search",
        qdrant_filters=filters_qdrant,  # 🔥 aplica filtros!
        alpha=0.7,  # 70% dense, 30% sparse
        top_k=3,
    )
    #  bm25_retriever = BM25Retriever.from_defaults(documents=[])

    # bm25_retriever = BM25Retriever.from_defaults(
    #      docstore=index.docstore, similarity_top_k=5
    # )

    # retriever = HybridRetriever(vector_retriever=vec_ret, bm25_retriever=bm25_retriever)

    # define custom retriever
    # keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index)
    # retriever = CustomHybridRetriever(vec_ret, bm25_retriever)
    # retriever = QueryFusionRetriever(
    #      [vec_ret, bm25_retriever],
    #      similarity_top_k=5,
    #     num_queries=1,  # set to 1 to disable query generation
    #      mode="relative_score",  # or "reciprocal_rerank"
    #     use_async=True,
    # )
    # configure response synthesizer
    # reranker = CohereRerank(api_key=cohere_api_key, top_n=2)
    # response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT)

    # Cria o sintetizador com citações ativadas
    #   response_synthesizer = get_response_synthesizer(
    ##       response_mode=ResponseMode.COMPACT, streaming=False
    #  )
    # assemble query engine
    query_engine = CitationQueryEngine.from_args(
        index=index,  # ✅ necessário!
        retriever=retriever,
        #  response_synthesizer=response_synthesizer,
        citation_chunk_size=512,
        citation_chunk_overlap=20,
        text_splitter=Settings.text_splitter,
        node_postprocessors=[CorrigirPageLabelPostprocessor()],
        # node_postprocessors=[reranker],
        # node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
        #  ],  # ,SimilarityPostprocessor(similarity_cutoff=0.7)],
    )

    query_engine.update_prompts(
        {
            "response_synthesizer:text_qa_template": text_qa_template,
            #   "response_synthesizer:refine_template": refine_template,
        }
    )

    return query_engine
