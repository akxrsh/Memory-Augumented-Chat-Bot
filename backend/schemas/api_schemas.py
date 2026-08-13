from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime

# --- Authentication & User Schemas ---
class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    role: str = Field(default="user")

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    username: str
    role: str

class UserOut(BaseModel):
    id: str
    username: str
    role: str
    created_at: datetime

    class Config:
        from_attributes = True


# --- Memory Schemas ---
class MemoryItem(BaseModel):
    id: str
    user_id: str
    text: str
    category: Optional[str] = "general"
    importance: int = Field(default=1, ge=1, le=10)
    created_at: datetime
    updated_at: datetime
    frequency: int = 1

    class Config:
        from_attributes = True

class MemoryAddRequest(BaseModel):
    text: str
    category: Optional[str] = "general"
    importance: Optional[int] = 1

class MemoryUpdateRequest(BaseModel):
    text: str
    category: Optional[str] = None
    importance: Optional[int] = None


# --- Knowledge Graph Schemas ---
class NodeSchema(BaseModel):
    id: str
    name: str
    label: str
    description: Optional[str] = None

class RelationshipSchema(BaseModel):
    id: str
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0

class GraphBuildRequest(BaseModel):
    text: str

class GraphQueryRequest(BaseModel):
    query: str


# --- Chat Schemas ---
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the sender (user, assistant, system, tool)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    stream: bool = False

class ChatResponse(BaseModel):
    response: str
    session_id: str
    retrieved_memories: List[str] = []
    retrieved_entities: List[str] = []
    retrieved_relations: List[str] = []
    sources: List[str] = []


# --- Document Upload & Scraping Schemas ---
class ScrapeRequest(BaseModel):
    url: str
    category: Optional[str] = "webpage"

class DocumentUploadResponse(BaseModel):
    filename: str
    chunks_inserted: int
    message: str


# --- Evaluation Schemas ---
class EvaluationRequest(BaseModel):
    session_id: Optional[str] = None
    user_query: str
    llm_response: str
    contexts: List[str] = []

class EvaluationResponse(BaseModel):
    faithfulness: float = Field(..., description="Groundedness score: 0 to 1")
    answer_relevance: float = Field(..., description="Relevance of the answer to user query: 0 to 1")
    context_recall: float = Field(..., description="Is context retrieved sufficient: 0 to 1")
    hallucination_rate: float = Field(..., description="Rate of hallucinated claims: 0 to 1")
    latency_seconds: float
    token_usage: Dict[str, int] = {}
    cost: float = 0.0
    report: str = ""
