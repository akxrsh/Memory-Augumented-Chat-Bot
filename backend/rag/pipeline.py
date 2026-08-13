import csv
import io
import time
import httpx
from pathlib import Path
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from backend.rag.vector_store import vector_store_manager
from backend.rag.text_cleaner import clean_html, clean_text, extract_metadata_from_soup
from backend.utils.logger import logger

class RAGPipeline:
    """Manages document parsing, scraping, chunking, indexing, and hybrid search (Dense + BM25)."""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        self.bm25_index: Optional[BM25Okapi] = None
        self.corpus_docs: List[Document] = []
        self._sync_bm25_from_vector_store()

    def _sync_bm25_from_vector_store(self):
        """Loads all documents from Chroma to build the BM25 index for sparse search."""
        try:
            if vector_store_manager.vector_store:
                collection = vector_store_manager.vector_store.get()
                documents = []
                if collection and "documents" in collection and collection["documents"]:
                    for idx, text in enumerate(collection["documents"]):
                        metadata = collection["metadatas"][idx] if collection["metadatas"] else {}
                        doc_id = collection["ids"][idx]
                        documents.append(Document(page_content=text, metadata={**metadata, "id": doc_id}))
                    
                    self.corpus_docs = documents
                    if documents:
                        tokenized_corpus = [doc.page_content.lower().split() for doc in documents]
                        self.bm25_index = BM25Okapi(tokenized_corpus)
                        logger.info(f"Synchronized BM25 index with {len(documents)} document chunks from Chroma.")
        except Exception as e:
            logger.error(f"Failed to synchronize BM25 index: {e}")

    async def ingest_file(self, file_content: bytes, file_name: str, file_type: str, metadata_override: Optional[Dict[str, Any]] = None) -> int:
        """Parses, chunks, and indexes a file based on its file extension type."""
        logger.info(f"Ingesting file: {file_name} ({file_type})")
        text = ""
        
        # 1. Parse content based on type
        if file_type.lower() == "pdf":
            try:
                pdf_reader = PdfReader(io.BytesIO(file_content))
                text_list = []
                num_pages = len(pdf_reader.pages)
                logger.info(f"PDF loaded successfully. Total pages: {num_pages}")
                
                # Limit parsing to first 30 pages for fast dry-runs
                max_pages = min(num_pages, 30)
                if num_pages > 30:
                    logger.info(f"Large PDF detected ({num_pages} pages). Limiting parsing to the first 30 pages for fast dry-run.")
                    
                for idx in range(max_pages):
                    page = pdf_reader.pages[idx]
                    if (idx + 1) % 10 == 0 or idx + 1 == max_pages:
                        logger.info(f"Parsing page {idx + 1}/{max_pages}...")
                    page_text = page.extract_text()
                    if page_text:
                        text_list.append(page_text)
                text = "\n\n".join(text_list)
                logger.info(f"Successfully extracted {len(text)} characters from PDF.")
            except Exception as e:
                logger.error(f"Error parsing PDF file: {e}")
                raise e
        elif file_type.lower() == "csv":
            try:
                text_stream = io.StringIO(file_content.decode("utf-8", errors="ignore"))
                reader = csv.reader(text_stream)
                rows = []
                headers = next(reader, None)
                for row in reader:
                    if headers:
                        row_str = ", ".join([f"{headers[i]}: {val}" for i, val in enumerate(row) if i < len(headers)])
                    else:
                        row_str = ", ".join(row)
                    rows.append(row_str)
                text = "\n".join(rows)
            except Exception as e:
                logger.error(f"Error parsing CSV file: {e}")
                raise e
        elif file_type.lower() in ["html", "htm"]:
            text = clean_html(file_content.decode("utf-8", errors="ignore"))
        elif file_type.lower() in ["md", "markdown", "txt"]:
            text = file_content.decode("utf-8", errors="ignore")
        else:
            # Try plain text fallback
            try:
                text = file_content.decode("utf-8")
            except Exception:
                raise ValueError(f"Unsupported file format: {file_type}")

        text = clean_text(text)
        if not text:
            logger.warning(f"No readable text extracted from file: {file_name}")
            return 0

        # 2. Chunking
        base_metadata = {
            "source": file_name,
            "file_type": file_type,
            "timestamp": time.time(),
            "title": file_name
        }
        if metadata_override:
            base_metadata.update(metadata_override)
            
        chunks = self.text_splitter.split_text(text)
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_index"] = i
            documents.append(Document(page_content=chunk, metadata=chunk_metadata))

        # 3. Add to Chroma
        added_count = await vector_store_manager.add_documents(documents)
        
        # 4. Refresh BM25 Index
        self._sync_bm25_from_vector_store()

        # 5. Extract and build graph in background (hybrid RAG + KG integration)
        try:
            from backend.graph.extractor import graph_extractor
            logger.info("Extracting Knowledge Graph nodes and relationships from ingested file...")
            await graph_extractor.extract_and_build(text)
        except Exception as ge_err:
            logger.error(f"Graph extraction failed during file ingestion: {ge_err}")

        return added_count

    async def scrape_and_ingest_url(self, url: str, metadata_override: Optional[Dict[str, Any]] = None) -> int:
        """Scrapes web page, extracts text, chunks it, and indexes it."""
        logger.info(f"Scraping and ingesting URL: {url}")
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                
            html_content = response.text
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Clean HTML & Extract Info
            body_text = clean_html(html_content)
            body_text = clean_text(body_text)
            
            web_metadata = extract_metadata_from_soup(soup, url)
            if metadata_override:
                web_metadata.update(metadata_override)
            
            if not body_text:
                logger.warning(f"Scraped empty body text from URL: {url}")
                return 0
                
            chunks = self.text_splitter.split_text(body_text)
            documents = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = web_metadata.copy()
                chunk_metadata["chunk_index"] = i
                chunk_metadata["timestamp"] = time.time()
                documents.append(Document(page_content=chunk, metadata=chunk_metadata))
                
            added_count = await vector_store_manager.add_documents(documents)
            self._sync_bm25_from_vector_store()

            # Extract and build graph in background (hybrid RAG + KG integration)
            try:
                from backend.graph.extractor import graph_extractor
                logger.info("Extracting Knowledge Graph nodes and relationships from scraped URL content...")
                await graph_extractor.extract_and_build(body_text)
            except Exception as ge_err:
                logger.error(f"Graph extraction failed during URL scrape ingestion: {ge_err}")

            return added_count
            
        except Exception as e:
            logger.error(f"Failed to scrape URL {url}: {e}")
            raise e

    async def hybrid_search(self, query: str, k: int = 4, rrf_constant: int = 60) -> List[Document]:
        """Performs Hybrid Search using Reciprocal Rank Fusion (RRF) on Dense and BM25 results."""
        logger.info(f"Starting hybrid search for query: '{query}'")
        
        # 1. Retrieve Dense (Vector) Results
        dense_results = await vector_store_manager.similarity_search(query, k=k*2)
        
        # 2. Retrieve BM25 (Sparse) Results
        sparse_results = []
        if self.bm25_index and self.corpus_docs:
            tokenized_query = query.lower().split()
            scores = self.bm25_index.get_scores(tokenized_query)
            # Zip scores with documents and sort descending
            scored_docs = sorted(zip(scores, self.corpus_docs), key=lambda x: x[0], reverse=True)
            # Take top docs that have positive score
            sparse_results = [doc for score, doc in scored_docs if score > 0][:k*2]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # Helper to generate unique key/id for ranking
        def get_doc_key(doc: Document) -> str:
            # Prefer unique ID if present, otherwise hash content
            doc_id = doc.metadata.get("id") or str(hash(doc.page_content))
            doc_map[doc_id] = doc
            return doc_id

        # Rank Dense results
        for rank, doc in enumerate(dense_results):
            doc_id = get_doc_key(doc)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_constant + rank + 1))

        # Rank Sparse results
        for rank, doc in enumerate(sparse_results):
            doc_id = get_doc_key(doc)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_constant + rank + 1))

        # Sort documents by RRF score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        final_results = [doc_map[key] for key in sorted_keys[:k]]
        logger.info(f"Hybrid search complete. Merged dense/sparse results and retrieved top {len(final_results)} documents.")
        return final_results

rag_pipeline = RAGPipeline()
