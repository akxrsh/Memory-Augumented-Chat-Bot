import os
import pytest
import asyncio
from datetime import datetime

# Configure env variables for testing before imports
os.environ["USE_LOCAL_FALLBACK"] = "True"
os.environ["LOCAL_DB_PATH"] = "data/test_fallback.db"
os.environ["CHROMA_DB_DIR"] = "data/test_chroma_db"

from backend.database.connection import db_manager
from backend.models.database import (
    db_create_user,
    db_get_user_by_username,
    db_add_memory,
    db_get_memories,
    db_add_graph_node,
    db_query_graph_nodes
)
from backend.services.auth import get_password_hash, verify_password, create_access_token
from backend.memory.manager import memory_manager
from backend.rag.pipeline import rag_pipeline
from backend.evaluation.evaluator import response_evaluator

@pytest.fixture(autouse=True)
async def setup_test_db():
    # Initialize connection
    await db_manager.initialize()
    yield
    # Cleanup DB connection and test DB file
    await db_manager.close()
    
    # Remove test SQLite DB
    if os.path.exists("data/test_fallback.db"):
        try:
            os.remove("data/test_fallback.db")
        except Exception:
            pass

@pytest.mark.asyncio
async def test_auth_and_user_creation():
    username = "test_user_99"
    raw_password = "securepassword123"
    hashed_pwd = get_password_hash(raw_password)
    
    # 1. Assert hashing works
    assert verify_password(raw_password, hashed_pwd) is True
    assert verify_password("wrongpassword", hashed_pwd) is False
    
    # 2. Assert registration inserts in SQLite
    user = await db_create_user(
        user_id="test-id-123",
        username=username,
        password_hash=hashed_pwd,
        role="user"
    )
    assert user["id"] == "test-id-123"
    assert user["username"] == username
    
    # 3. Assert lookup succeeds
    lookup = await db_get_user_by_username(username)
    assert lookup is not None
    assert lookup["id"] == "test-id-123"

@pytest.mark.asyncio
async def test_user_memory_deduplication():
    user_id = "test-user-uuid"
    
    # 1. Add fresh fact
    fact1 = await db_add_memory(user_id, "User works as a senior dev", "profession", 7)
    assert fact1["text"] == "User works as a senior dev"
    
    # 2. Get facts
    facts = await db_get_memories(user_id)
    assert len(facts) >= 1
    assert any(f["text"] == "User works as a senior dev" for f in facts)
    
    # 3. Test manager compilation
    profile = await memory_manager.compile_user_profile(user_id)
    assert "senior dev" in profile

@pytest.mark.asyncio
async def test_knowledge_graph_sqlite_fallback():
    # 1. Save Concept Node
    node = await db_add_graph_node(name="LangGraph", label="Framework", description="Graph routing agentic core")
    assert node.name == "LangGraph"
    assert node.label == "Framework"
    
    # 2. Query nodes
    results = await db_query_graph_nodes("Lang")
    assert len(results) >= 1
    assert any(r["name"] == "LangGraph" for r in results)

@pytest.mark.asyncio
async def test_rag_pipeline_parsing():
    content = b"Memory-Augmented Chatbots scale retrieval using dense models like Sentence Transformers."
    file_name = "test_doc.txt"
    
    # Ingest text document
    chunks_inserted = await rag_pipeline.ingest_file(
        file_content=content,
        file_name=file_name,
        file_type="txt"
    )
    assert chunks_inserted >= 1
    
    # Hybrid search retrieval
    search_results = await rag_pipeline.hybrid_search("dense models", k=1)
    assert len(search_results) >= 1
    assert "Sentence Transformers" in search_results[0].page_content

@pytest.mark.asyncio
async def test_evaluator_metrics():
    query = "Explain LangGraph"
    response = "LangGraph is a stateful agent orchestration library that compiles actions as nodes and edges."
    contexts = ["LangGraph is a framework for constructing cyclic states and graphs for LLM agents."]
    
    report = await response_evaluator.evaluate_response(query, response, contexts)
    
    assert "faithfulness" in report
    assert "answer_relevance" in report
    assert "context_recall" in report
    assert report["latency"] > 0.0
    assert report["cost"] >= 0.0
