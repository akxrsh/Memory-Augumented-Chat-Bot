from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any
from backend.schemas.api_schemas import GraphBuildRequest, GraphQueryRequest
from backend.graph.extractor import graph_extractor
from backend.models.database import db_query_graph_nodes, db_get_graph_neighborhood
from backend.services.auth import get_current_user
from backend.utils.logger import logger

router = APIRouter(prefix="/graph", tags=["Knowledge Graph"])

@router.post("/build", status_code=status.HTTP_201_CREATED)
async def build_graph_from_text(
    payload: GraphBuildRequest,
    current_user: dict = Depends(get_current_user)
):
    """Processes text using an LLM to extract nodes and relationships, then inserts them into the Knowledge Graph."""
    logger.info(f"User {current_user['username']} is trigger graph extraction.")
    try:
        result = await graph_extractor.extract_and_build(payload.text)
        return {
            "status": "success",
            "extracted_nodes_count": len(result["nodes"]),
            "extracted_relationships_count": len(result["relationships"]),
            "data": result
        }
    except Exception as e:
        logger.error(f"Graph build failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process text: {str(e)}"
        )

@router.get("/query")
async def query_nodes(
    query: str,
    current_user: dict = Depends(get_current_user)
):
    """Queries nodes matching a substring name search in the graph database."""
    try:
        nodes = await db_query_graph_nodes(query)
        return nodes
    except Exception as e:
        logger.error(f"Graph node query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/neighborhood")
async def get_neighborhood(
    node_names: str,  # Comma separated node names
    current_user: dict = Depends(get_current_user)
):
    """Retrieves adjacent nodes and relations for a list of node names (useful for force-directed graph UI)."""
    names_list = [name.strip() for name in node_names.split(",") if name.strip()]
    if not names_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide at least one node name in comma-separated list."
        )
        
    try:
        # Dynamic self-healing visualization fallback:
        # If frontend requests default nodes, dynamically retrieve the top 20 nodes from the DB.
        if set(names_list) == {"LangGraph", "LangChain", "RAG"}:
            all_nodes = await db_query_graph_nodes("")
            if all_nodes:
                names_list = [n["name"] for n in all_nodes[:20]]
                
        neighborhood = await db_get_graph_neighborhood(names_list)
        return neighborhood
    except Exception as e:
        logger.error(f"Graph neighborhood retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
