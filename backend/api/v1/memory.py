from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any
from backend.schemas.api_schemas import MemoryAddRequest, MemoryUpdateRequest, MemoryItem
from backend.memory.manager import memory_manager
from backend.models.database import db_get_memories, db_delete_memory, db_add_memory
from backend.services.auth import get_current_user
from backend.utils.logger import logger

router = APIRouter(prefix="/memory", tags=["Long-Term Memory"])

@router.post("/add", response_model=MemoryItem, status_code=status.HTTP_201_CREATED)
async def add_memory_manually(
    payload: MemoryAddRequest,
    current_user: dict = Depends(get_current_user)
):
    """Manually registers a long-term fact about the user."""
    user_id = current_user["id"]
    logger.info(f"User {current_user['username']} is adding a manual memory fact.")
    try:
        # Pre-validate inputs
        importance = payload.importance or 5
        category = payload.category or "general"
        
        # Save memory and index it
        mem_doc = await memory_manager._merge_or_add_fact(
            user_id=user_id,
            new_text=payload.text,
            category=category,
            importance=importance
        )
        return mem_doc
    except Exception as e:
        logger.error(f"Failed to add memory: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("", response_model=List[MemoryItem])
async def get_all_user_memories(
    current_user: dict = Depends(get_current_user)
):
    """Retrieves all memory facts stored for the current authenticated user."""
    user_id = current_user["id"]
    try:
        memories = await db_get_memories(user_id)
        return memories
    except Exception as e:
        logger.error(f"Failed to query user memories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.delete("/{memory_id}", status_code=status.HTTP_200_OK)
async def delete_user_memory(
    memory_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Deletes a specific memory fact by its ID."""
    logger.info(f"User {current_user['username']} is deleting memory ID: {memory_id}")
    try:
        # Check if memory belongs to user (for security/isolation)
        memories = await db_get_memories(current_user["id"])
        belongs_to_user = any(m["id"] == memory_id for m in memories)
        
        if not belongs_to_user:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to delete this memory."
            )
            
        success = await db_delete_memory(memory_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Memory fact not found."
            )
            
        return {"status": "success", "message": "Memory fact deleted successfully."}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Failed to delete memory: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/extract")
async def extract_memories_manually(
    text: str,
    current_user: dict = Depends(get_current_user)
):
    """Analyzes text for testing purposes to see what memory facts would be extracted and saved."""
    user_id = current_user["id"]
    try:
        extracted = await memory_manager.extract_and_store_memory(user_id, text)
        return {
            "status": "success",
            "extracted_count": len(extracted),
            "data": extracted
        }
    except Exception as e:
        logger.error(f"Manual extraction test failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
