import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

from backend.config.settings import settings
from backend.services.llm import llm_manager
from backend.rag.vector_store import vector_store_manager
from backend.models.database import (
    db_add_memory,
    db_get_memories,
    db_update_memory,
    db_delete_memory
)
from backend.utils.logger import logger

class MemoryManager:
    """Manages the user long-term memory system (MongoDB/SQLite + Chroma embeddings)."""

    def __init__(self):
        self.llm = llm_manager.llm

    async def extract_and_store_memory(self, user_id: str, text: str) -> List[Dict[str, Any]]:
        """Analyzes text for user facts (e.g. name, preferences, goals) and updates memory store."""
        logger.info(f"Extracting memory facts for user: {user_id}...")
        
        system_prompt = (
            "You are a long-term memory extraction assistant. Analyze the user's message and extract permanent facts "
            "about them (their name, age, job, skills, interests, goals, preferences, favorite technologies).\n"
            "Only extract statements of long-term importance. Do not extract temporary status updates (e.g., 'I am hungry today').\n\n"
            "Respond ONLY with a valid JSON array of objects. Do not include explanation or Markdown codeblocks.\n\n"
            "JSON Format:\n"
            "[\n"
            "  {\n"
            "    \"text\": \"User lives in San Francisco\",\n"
            "    \"category\": \"preferences | profession | background | interests | goals | tech\",\n"
            "    \"importance\": 7\n"
            "  }\n"
            "]\n\n"
            "Scale importance from 1 (very minor preference) to 10 (user name, core identity, critical goal)."
        )
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Message to analyze: {text}")
            ])
            
            raw_content = response.content.strip()
            if raw_content.startswith("```"):
                raw_content = re.sub(r"^```(?:json)?\n", "", raw_content)
                raw_content = re.sub(r"\n```$", "", raw_content)
                raw_content = raw_content.strip()
                
            # If empty or mock response
            if not raw_content or raw_content.startswith("This is a fallback"):
                return []
                
            extracted_facts = json.loads(raw_content)
            if not isinstance(extracted_facts, list):
                extracted_facts = [extracted_facts]
                
            stored_memories = []
            for fact in extracted_facts:
                fact_text = fact.get("text", "").strip()
                category = fact.get("category", "general").strip()
                importance = int(fact.get("importance", 1))
                
                if fact_text:
                    saved_mem = await self._merge_or_add_fact(user_id, fact_text, category, importance)
                    stored_memories.append(saved_mem)
                    
            return stored_memories
            
        except Exception as e:
            logger.error(f"Memory extraction failed: {e}")
            return []

    async def _merge_or_add_fact(self, user_id: str, new_text: str, category: str, importance: int) -> Dict[str, Any]:
        """Resolves conflicts, updates frequency, or adds a new memory fact to databases."""
        existing_memories = await db_get_memories(user_id)
        
        # Check semantic similarity using LLM or Chroma.
        # For simplicity and robust conflict resolution, we query the LLM to check if there is an overlapping fact.
        conflict_system = (
            "You are a memory deduplication engine. Check if the new fact conflicts with, merges with, or is completely "
            "independent from any of the existing facts.\n\n"
            "Existing Facts:\n"
            "{existing_list}\n\n"
            "New Fact: {new_fact}\n\n"
            "Return a JSON object indicating the action to take:\n"
            "{\n"
            "  \"action\": \"create\" | \"update\" | \"noop\",\n"
            "  \"target_id\": \"ID of the existing fact (if update or noop)\",\n"
            "  \"merged_text\": \"Fully compiled updated fact incorporating both new details (if update)\"\n"
            "}"
        )
        
        existing_list_str = "\n".join([f"- [{m['id']}] ({m['category']}): {m['text']}" for m in existing_memories])
        
        action = "create"
        target_id = None
        merged_text = new_text
        
        if existing_memories:
            try:
                check_prompt = conflict_system.format(existing_list=existing_list_str, new_fact=new_text)
                res = await self.llm.ainvoke([SystemMessage(content=check_prompt)])
                
                res_content = res.content.strip()
                if res_content.startswith("```"):
                    res_content = re.sub(r"^```(?:json)?\n", "", res_content)
                    res_content = re.sub(r"\n```$", "", res_content)
                    res_content = res_content.strip()
                    
                resolution = json.loads(res_content)
                action = resolution.get("action", "create")
                target_id = resolution.get("target_id")
                merged_text = resolution.get("merged_text", new_text)
            except Exception as e:
                logger.warning(f"Memory conflict check failed, defaulting to create: {e}")
                action = "create"

        if action == "update" and target_id:
            # Find the original record to update frequency
            orig = next((m for m in existing_memories if m["id"] == target_id), None)
            freq = (orig["frequency"] + 1) if orig else 2
            
            updates = {
                "text": merged_text,
                "importance": max(importance, orig["importance"] if orig else importance),
                "frequency": freq
            }
            await db_update_memory(target_id, updates)
            
            # Re-index in Chroma (first delete old chunk if exists)
            await self._index_memory_in_chroma(user_id, target_id, merged_text, category)
            
            logger.info(f"Updated memory [{target_id}]: {merged_text}")
            return {**orig, **updates} if orig else {"id": target_id, "text": merged_text}
            
        elif action == "noop" and target_id:
            # Simple frequency increase
            orig = next((m for m in existing_memories if m["id"] == target_id), None)
            if orig:
                await db_update_memory(target_id, {"frequency": orig["frequency"] + 1})
            return orig or {}
            
        else:
            # Create new memory
            mem_doc = await db_add_memory(user_id, new_text, category, importance)
            await self._index_memory_in_chroma(user_id, mem_doc["id"], new_text, category)
            logger.info(f"Created new memory: {new_text}")
            return mem_doc

    async def _index_memory_in_chroma(self, user_id: str, fact_id: str, text: str, category: str):
        """Indexes/re-indexes memory fact into Chroma vector database for semantic retrieval."""
        doc = Document(
            page_content=text,
            metadata={
                "id": fact_id,
                "user_id": user_id,
                "type": "memory",
                "category": category
            }
        )
        try:
            # Add to Chroma (uses existing collection)
            await vector_store_manager.add_documents([doc])
        except Exception as e:
            logger.error(f"Failed to index memory in Chroma: {e}")

    async def get_semantic_memories(self, user_id: str, query: str, k: int = 5) -> List[str]:
        """Queries Chroma for user's long-term memory facts using semantic search."""
        try:
            # Chroma filters on metadata
            filter_dict = {
                "$and": [
                    {"user_id": {"$eq": user_id}},
                    {"type": {"$eq": "memory"}}
                ]
            }
            docs = await vector_store_manager.similarity_search(query, k=k, filter_dict=filter_dict)
            return [d.page_content for d in docs]
        except Exception as e:
            logger.error(f"Semantic memory query failed: {e}")
            # Fallback to text matching
            all_mem = await db_get_memories(user_id)
            return [m["text"] for m in all_mem[:k]]

    async def compile_user_profile(self, user_id: str) -> str:
        """Summarizes the user's memory facts into a clean system prompt profile."""
        memories = await db_get_memories(user_id)
        if not memories:
            return "No prior user details known."
            
        profile_parts = []
        for m in memories:
            profile_parts.append(f"- {m['text']} (Category: {m['category']}, Relevance/Freq: {m['frequency']})")
            
        return "\n".join(profile_parts)

memory_manager = MemoryManager()
