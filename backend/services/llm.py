from typing import Optional, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from backend.config.settings import settings
from backend.utils.logger import logger

class LocalMockLLM(BaseChatModel):
    """A local mock LLM that generates responses based on heuristics and context. Used when API keys are absent."""
    
    model_name: str = "mock-local"
    
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs) -> Any:
        last_message = messages[-1].content.lower() if messages else ""
        system_content = "\n".join([m.content for m in messages if isinstance(m, SystemMessage)])
        
        # Build smart local responses based on context and query
        if "extract" in system_content.lower() or "entities" in last_message:
            text = (
                '{\n  "entities": [\n    {"name": "LangGraph", "label": "Framework", "description": "Graph-based LLM orchestrator"},\n'
                '    {"name": "LangChain", "label": "Ecosystem", "description": "LLM tool suite"}\n  ],\n'
                '  "relationships": [\n    {"source": "LangGraph", "target": "LangChain", "relation_type": "BELONGS_TO", "weight": 1.0}\n  ]\n}'
            )
        elif "deduplication" in system_content.lower() or "conflict" in system_content.lower():
            text = '{\n  "action": "create",\n  "target_id": null,\n  "merged_text": ""\n}'
        elif "hallucination judge" in system_content.lower() or "faithfulness" in system_content.lower():
            text = '{\n  "passed": true,\n  "reason": ""\n}'
        elif "companies use langgraph" in last_message:
            text = (
                "Based on the Knowledge Graph, several leading AI companies use LangGraph for orchestration. "
                "Specifically, LangChain Inc., Google DeepMind, and Anthropic use LangGraph in "
                "their agentic development workflows."
            )
        elif any(g in last_message for g in ["hello", "hi", "hey", "who are you", "what can you do"]):
            text = (
                "Hello! I am your Memory-Augmented AI Assistant. I can track long-term facts about your goals, "
                "perform hybrid document retrieval (RAG), query knowledge graph relationships, and execute tools.\n\n"
                "*(Running in Local Offline Mode. To connect real GPT-4o or Groq models, add an `OPENAI_API_KEY` or `GROQ_API_KEY` to your `.env` file.)*"
            )
        elif "remember" in last_message or "my name" in last_message or "who am i" in last_message:
            text = (
                "Yes! I track all your details in your Memory Vault. Check the 'User Profile' tab on the left sidebar to see all stored facts about your background and preferences."
            )
        else:
            # If retrieved context is available in system message, use it
            if "RETRIEVED DOCUMENT DATA" in system_content and "No RAG documents retrieved" not in system_content:
                text = f"Based on the documents in your ingestion pipeline:\n\n{system_content.split('--- RETRIEVED DOCUMENT DATA (RAG) ---')[1].split('---')[0].strip()}"
            else:
                text = (
                    f"I have processed your query: '{messages[-1].content}'.\n\n"
                    "*(Note: System running in Offline Demo Mode. To get live AI answers from GPT-4o-mini or Groq LLaMA, set your `OPENAI_API_KEY` or `GROQ_API_KEY` in the `.env` file.)*"
                )
            
        ai_message = AIMessage(content=text)
        from langchain_core.outputs import ChatGeneration, ChatResult
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    def _llm_type(self) -> str:
        return "mock-local"

class LLMManager:
    """Manages chat model instances, providing a clean fallback if keys are missing."""

    def __init__(self):
        self.llm = self._init_llm()

    def _init_llm(self) -> BaseChatModel:
        if settings.has_valid_openai_key:
            logger.info(f"Initializing ChatOpenAI model: {settings.LLM_MODEL}...")
            return ChatOpenAI(
                openai_api_key=settings.OPENAI_API_KEY,
                model=settings.LLM_MODEL,
                temperature=0.0
            )
        elif settings.has_valid_groq_key:
            logger.info(f"Initializing ChatGroq model: {settings.GROQ_MODEL}...")
            from langchain_groq import ChatGroq
            return ChatGroq(
                groq_api_key=settings.GROQ_API_KEY,
                model=settings.GROQ_MODEL,
                temperature=0.0
            )
        else:
            logger.warning("No OPENAI_API_KEY or GROQ_API_KEY found. Initializing Local Mock LLM for dry-runs...")
            return LocalMockLLM()

llm_manager = LLMManager()
