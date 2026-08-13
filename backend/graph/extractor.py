import json
import re
from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from backend.services.llm import llm_manager
from backend.models.database import db_add_graph_node, db_add_graph_relationship
from backend.utils.logger import logger

class GraphExtractor:
    """Uses LLM to extract nodes (entities) and edges (relations) and write them to the graph DB."""

    @property
    def llm(self):
        return llm_manager.llm

    async def extract_and_build(self, text: str) -> Dict[str, Any]:
        """Extracts concepts and relationships from raw text, then writes them to the graph database."""
        max_chars = 15000
        if len(text) > max_chars:
            logger.info(f"Text length ({len(text)} chars) exceeds maximum threshold. Truncating to first {max_chars} chars for graph extraction.")
            text = text[:max_chars]

        logger.info("Extracting entities and relationships from text...")
        
        system_prompt = (
            "You are a knowledge graph builder. Extract entities (concepts, technologies, companies, frameworks, etc.) "
            "and their relationships from the user's text.\n\n"
            "Respond ONLY with a valid JSON object matching the schema below. Do not include explanation or Markdown codeblocks.\n\n"
            "JSON Schema:\n"
            "{\n"
            "  \"entities\": [\n"
            "    {\"name\": \"Entity Name\", \"label\": \"Concept or Technology or Framework\", \"description\": \"Brief description of what this entity is\"}\n"
            "  ],\n"
            "  \"relationships\": [\n"
            "    {\"source\": \"Source Entity Name\", \"target\": \"Target Entity Name\", \"relation_type\": \"BELONGS_TO or RELATED_TO or DEVELOPED_BY etc.\", \"weight\": 1.0}\n"
            "  ]\n"
            "}"
        )
        
        extracted_nodes = []
        extracted_rels = []
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Extract entities and relationships from this text:\n\n{text}")
            ])
            
            raw_content = response.content.strip()
            if raw_content.startswith("```"):
                raw_content = re.sub(r"^```(?:json)?\n", "", raw_content)
                raw_content = re.sub(r"\n```$", "", raw_content)
                raw_content = raw_content.strip()
                
            data = json.loads(raw_content)
            extracted_nodes = data.get("entities", [])
            extracted_rels = data.get("relationships", [])
        except Exception as e:
            logger.warning(f"LLM entity extraction fallback: {e}. Running keyword entity extraction...")
            words = re.findall(r'\b[A-Z][a-zA-Z0-9_\-\.]{2,}\b', text)
            stopwords = {"The", "This", "That", "There", "Here", "With", "From", "Have", "Were", "Been", "When", "What", "Where", "Which", "Your", "User", "Page", "HTTP", "HTTPS", "HTML", "Copyright"}
            unique_words = list(dict.fromkeys([w for w in words if w not in stopwords]))[:10]
            
            if len(unique_words) >= 2:
                for w in unique_words:
                    extracted_nodes.append({"name": w, "label": "Concept", "description": f"Extracted concept: {w}"})
                for i in range(len(unique_words) - 1):
                    extracted_rels.append({
                        "source": unique_words[i],
                        "target": unique_words[i+1],
                        "relation_type": "RELATED_TO",
                        "weight": 1.0
                    })
            elif unique_words:
                extracted_nodes.append({"name": unique_words[0], "label": "Concept", "description": f"Extracted concept: {unique_words[0]}"})

        added_nodes = []
        added_rels = []
        
        # Save nodes
        for node in extracted_nodes:
            name = node.get("name")
            label = node.get("label", "Concept")
            description = node.get("description", "")
            if name:
                node_obj = await db_add_graph_node(name=name, label=label, description=description)
                added_nodes.append(node_obj.dict())
                
        # Save relationships
        for rel in extracted_rels:
            source = rel.get("source")
            target = rel.get("target")
            rel_type = rel.get("relation_type", "RELATED_TO").upper().replace(" ", "_")
            weight = float(rel.get("weight", 1.0))
            if source and target:
                rel_obj = await db_add_graph_relationship(
                    source_name=source,
                    target_name=target,
                    relation_type=rel_type,
                    weight=weight
                )
                added_rels.append(rel_obj.dict())
                
        logger.info(f"Graph extraction finished. Added {len(added_nodes)} nodes and {len(added_rels)} relationships.")
        return {
            "nodes": added_nodes,
            "relationships": added_rels
        }

graph_extractor = GraphExtractor()
