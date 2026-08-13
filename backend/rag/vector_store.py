import os
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

from backend.config.settings import settings
from backend.utils.logger import logger

class VectorStoreManager:
    """Manages Vector Store (Chroma) initialization, embedding loading, and search."""

    def __init__(self):
        self.embeddings = None
        self.vector_store: Optional[Chroma] = None
        try:
            self.embeddings = self._init_embeddings()
            self._init_vector_store()
        except Exception as e:
            logger.warning(f"Initial Chroma vector store setup postponed: {e}")

    def _ensure_initialized(self):
        """Lazy loader to guarantee Chroma database is booted before operations."""
        if self.vector_store is None:
            logger.info("Initializing vector store dynamically...")
            try:
                if not self.embeddings:
                    self.embeddings = self._init_embeddings()
                self._init_vector_store()
            except Exception as e:
                logger.error(f"Dynamic vector store initialization failed: {e}")
                raise e

    def _init_embeddings(self):
        """Initializes OpenAI embeddings if valid key is present, otherwise falls back to FakeEmbeddings."""
        if settings.has_valid_openai_key:
            logger.info("Initializing OpenAI Embeddings...")
            return OpenAIEmbeddings(
                openai_api_key=settings.OPENAI_API_KEY,
                model=settings.EMBEDDING_MODEL
            )
        else:
            logger.info("No valid OpenAI API Key found. Initializing local FakeEmbeddings fallback...")
            from langchain_core.embeddings import FakeEmbeddings
            return FakeEmbeddings(size=1536)

    def _init_vector_store(self):
        """Initializes the persistent Chroma database client."""
        db_dir = Path(settings.CHROMA_DB_DIR)
        db_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing persistent Chroma vector store at: {db_dir.resolve()}")
        try:
            self.vector_store = Chroma(
                persist_directory=str(db_dir),
                embedding_function=self.embeddings,
                collection_name="chatbot_kb"
            )
            logger.info("Chroma vector store initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing Chroma vector store: {e}")
            raise e

    async def add_documents(self, documents: List[Document]) -> int:
        """Adds LangChain documents to the vector store and persists changes."""
        self._ensure_initialized()
        if self.vector_store is None:
            raise ValueError("Vector store is not initialized.")
        
        if not documents:
            return 0

        logger.info(f"Adding {len(documents)} documents to vector store...")
        try:
            self.vector_store.add_documents(documents)
            logger.info("Documents added and database updated.")
            return len(documents)
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            raise e

    async def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Searches vector store for top-k similar documents, optionally filtering on metadata."""
        self._ensure_initialized()
        if self.vector_store is None:
            raise ValueError("Vector store is not initialized.")
            
        logger.info(f"Performing vector similarity search for query: '{query}', k={k}, filters={filter_dict}")
        try:
            results = self.vector_store.similarity_search(query, k=k, filter=filter_dict)
            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def delete_all(self):
        """Resets the vector collection."""
        if self.vector_store is None:
            return
        logger.warning("Resetting the entire vector store database...")
        try:
            self.vector_store.delete_collection()
            self._init_vector_store()  # Re-initialize empty
            logger.info("Vector store reset complete.")
        except Exception as e:
            logger.error(f"Failed to reset vector store: {e}")

vector_store_manager = VectorStoreManager()
