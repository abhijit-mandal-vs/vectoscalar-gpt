from typing import Optional
from pathlib import Path
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.chain_filter import LLMChainFilter
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_chroma import Chroma

from vsgpt.config import Config
from vsgpt.model import create_embeddings, create_reranker

def create_retriever(
    llm: BaseLanguageModel, vector_store: Optional[VectorStore] = None
) -> VectorStoreRetriever:
    embedding = create_embeddings()
    
    if vector_store is None:
        # Create a new Chroma instance if no vector_store is provided
        vector_store = Chroma(
            embedding_function=embedding,
            collection_name=Config.Database.DOCUMENTS_COLLECTION,
            persist_directory=Config.Path.DATABASE_DIR,
        )
    elif isinstance(vector_store, (str, Path)):
        # Load the vector store from the given path
        vector_store = Chroma(
            persist_directory=str(vector_store),
            embedding_function=embedding,
            collection_name=Config.Database.DOCUMENTS_COLLECTION,
        )

    # Now vector_store should always be a VectorStore instance
    retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 5, "fetch_k": 50}
    )

    if Config.Retriever.USE_RERANKER:
        retriever = ContextualCompressionRetriever(
            base_compressor=create_reranker(), base_retriever=retriever
        )

    if Config.Retriever.USE_CHAIN_FILTER:
        retriever = ContextualCompressionRetriever(
            base_compressor=LLMChainFilter.from_llm(llm), base_retriever=retriever
        )

    return retriever