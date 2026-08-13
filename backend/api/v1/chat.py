import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from backend.schemas.api_schemas import ChatRequest, ChatResponse
from backend.services.agent import compiled_agent
from backend.memory.manager import memory_manager
from backend.evaluation.evaluator import response_evaluator
from backend.models.database import db_get_or_create_conversation, db_save_conversation, db_get_all_user_conversations
from backend.services.auth import get_current_user
from backend.utils.logger import logger

router = APIRouter(prefix="/chat", tags=["Agent Chat & Orchestration"])

@router.post("", response_model=ChatResponse)
async def chat_interaction(
    payload: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Interacts with the stateful LangGraph agent and records chat session history."""
    user_id = current_user["id"]
    session_id = payload.session_id or str(uuid.uuid4())
    
    logger.info(f"Chat request received for user: {current_user['username']}, session: {session_id}")
    
    try:
        # 1. Fetch or initialize chat history
        conversation = await db_get_or_create_conversation(session_id, user_id)
        
        # 2. Run LangGraph Workflow
        initial_state = {
            "messages": conversation.get("messages", []),
            "user_id": user_id,
            "session_id": session_id,
            "user_query": payload.message,
            "memory_profile": "",
            "relevant_memories": [],
            "intent": "DIRECT_CHAT",
            "retrieved_contexts": [],
            "graph_nodes": [],
            "graph_rels": [],
            "tool_outputs": [],
            "final_response": "",
            "evaluation_passed": True,
            "retry_count": 0,
            "feedback": ""
        }
        
        logger.info(f"Running LangGraph agent workflow for session {session_id}...")
        result = await compiled_agent.ainvoke(initial_state)
        
        response_text = result["final_response"]
        
        # 3. Update Conversation History in Database
        updated_messages = conversation.get("messages", [])
        # Append User Message
        updated_messages.append({
            "role": "user",
            "content": payload.message,
            "timestamp": datetime.utcnow().isoformat()
        })
        # Append Assistant Response
        updated_messages.append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Set dynamic chat title if it's the first message
        title = None
        if len(updated_messages) <= 2:
            title = payload.message[:30] + "..." if len(payload.message) > 30 else payload.message
            
        await db_save_conversation(
            session_id=session_id,
            messages=updated_messages,
            title=title
        )
        
        # 4. Trigger memory extraction in the background (asynchronous & non-blocking)
        background_tasks.add_task(
            memory_manager.extract_and_store_memory,
            user_id=user_id,
            text=payload.message
        )
        
        # 5. Trigger offline Ragas evaluation in the background (asynchronous & non-blocking)
        background_tasks.add_task(
            response_evaluator.evaluate_response,
            query=payload.message,
            response=response_text,
            contexts=result.get("retrieved_contexts") or []
        )
        
        return {
            "response": response_text,
            "session_id": session_id,
            "retrieved_memories": result.get("relevant_memories", []),
            "retrieved_entities": result.get("graph_nodes", []),
            "retrieved_relations": result.get("graph_rels", []),
            "sources": list(set(result.get("retrieved_contexts", [])))
        }
        
    except Exception as e:
        logger.error(f"Chat interaction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent orchestration error: {str(e)}"
        )

@router.get("/sessions")
async def get_chat_sessions(
    current_user: dict = Depends(get_current_user)
):
    """Retrieves all active and historical chat session titles for the current user."""
    try:
        sessions = await db_get_all_user_conversations(current_user["id"])
        return sessions
    except Exception as e:
        logger.error(f"Failed to fetch chat sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/sessions/{session_id}")
async def get_session_messages(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Retrieves all chat messages for a specific session ID."""
    try:
        conversation = await db_get_or_create_conversation(session_id, current_user["id"])
        return conversation
    except Exception as e:
        logger.error(f"Failed to fetch session messages: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
