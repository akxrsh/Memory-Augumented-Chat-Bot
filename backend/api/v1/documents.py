from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from typing import List, Optional
from backend.schemas.api_schemas import ScrapeRequest, DocumentUploadResponse
from backend.rag.pipeline import rag_pipeline
from backend.services.auth import get_current_user, require_role
from backend.utils.logger import logger

router = APIRouter(prefix="/documents", tags=["Document Ingestion & RAG"])

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    category: Optional[str] = Form("general"),
    current_user: dict = Depends(require_role(["admin", "user"]))
):
    """Uploads and indexes a document (PDF, MD, CSV, HTML, TXT)."""
    logger.info(f"User {current_user['username']} is uploading document: {file.filename}")
    
    file_extension = file.filename.split(".")[-1].lower()
    allowed_extensions = ["pdf", "csv", "md", "markdown", "txt", "html", "htm"]
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension: .{file_extension}. Allowed extensions: {allowed_extensions}"
        )
        
    try:
        content = await file.read()
        chunks_inserted = await rag_pipeline.ingest_file(
            file_content=content,
            file_name=file.filename,
            file_type=file_extension,
            metadata_override={
                "category": category,
                "uploaded_by": current_user["username"]
            }
        )
        return {
            "filename": file.filename,
            "chunks_inserted": chunks_inserted,
            "message": f"Successfully parsed and indexed {chunks_inserted} chunk(s) from document."
        }
    except Exception as e:
        logger.error(f"Failed to ingest file {file.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index document: {str(e)}"
        )

@router.post("/scrape", response_model=DocumentUploadResponse)
async def scrape_url(
    payload: ScrapeRequest,
    current_user: dict = Depends(require_role(["admin", "user"]))
):
    """Scrapes content from a URL and indexes it."""
    logger.info(f"User {current_user['username']} is scraping URL: {payload.url}")
    try:
        chunks_inserted = await rag_pipeline.scrape_and_ingest_url(
            url=payload.url,
            metadata_override={
                "category": payload.category,
                "scraped_by": current_user["username"]
            }
        )
        return {
            "filename": payload.url,
            "chunks_inserted": chunks_inserted,
            "message": f"Successfully scraped and indexed {chunks_inserted} chunk(s) from website."
        }
    except Exception as e:
        logger.error(f"Failed to scrape URL {payload.url}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to scrape and index URL: {str(e)}"
        )

@router.post("/search")
async def test_search(
    query: str,
    k: int = 4,
    current_user: dict = Depends(get_current_user)
):
    """Test endpoint for performing Hybrid search and viewing retrieved chunks."""
    try:
        results = await rag_pipeline.hybrid_search(query, k=k)
        return [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            } for doc in results
        ]
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.delete("/reset", status_code=status.HTTP_200_OK)
async def reset_vector_store(
    current_user: dict = Depends(require_role(["admin"]))
):
    """Resets the vector database collections (Admin only)."""
    logger.warning(f"Admin {current_user['username']} is resetting the vector store.")
    try:
        from backend.rag.vector_store import vector_store_manager
        await vector_store_manager.delete_all()
        # Reset RAG Pipeline BM25 corpus as well
        rag_pipeline.corpus_docs = []
        rag_pipeline.bm25_index = None
        return {"status": "success", "message": "Vector store and BM25 index reset successfully."}
    except Exception as e:
        logger.error(f"Vector database reset failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset vector database: {str(e)}"
        )
