import json
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from backend.config.settings import settings
from backend.services.llm import llm_manager
from backend.memory.manager import memory_manager
from backend.rag.pipeline import rag_pipeline
from backend.models.database import db_query_graph_nodes, db_get_graph_neighborhood
from backend.tools.custom_tools import ALL_TOOLS, TOOL_MAP
from backend.utils.logger import logger

# Define the State Schema
class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    user_id: str
    session_id: str
    user_query: str
    memory_profile: str
    relevant_memories: List[str]
    intent: str
    retrieved_contexts: List[str]
    graph_nodes: List[str]
    graph_rels: List[str]
    tool_outputs: List[str]
    final_response: str
    evaluation_passed: bool
    retry_count: int
    feedback: str

# ==========================================
# GRAPH NODE FUNCTIONS
# ==========================================

async def retrieve_user_context(state: AgentState) -> Dict[str, Any]:
    """Retrieves long-term user memories and compiles a profile context."""
    user_id = state["user_id"]
    query = state["user_query"]
    
    logger.info(f"[Node: retrieve_user_context] Retrieving memory for user {user_id}")
    
    profile = await memory_manager.compile_user_profile(user_id)
    relevant = await memory_manager.get_semantic_memories(user_id, query, k=3)
    
    return {
        "memory_profile": profile,
        "relevant_memories": relevant
    }

async def classify_intent(state: AgentState) -> Dict[str, Any]:
    """Uses LLM to classify user message intent."""
    query = state["user_query"]
    logger.info(f"[Node: classify_intent] Classifying query: '{query}'")
    
    system_prompt = (
        "You are an intent detection router. Classify the user query into one of these four categories:\n"
        "1. KNOWLEDGE_GRAPH: Use this for requests asking about connections, ecosystems, links, hierarchies, or structures (e.g. 'what companies use X?', 'what frameworks belong to ecosystem Y?').\n"
        "2. RAG: Use this for queries about static articles, documents, concepts, codes, or uploaded files (e.g. 'explain how RAG works', 'what does document X say?').\n"
        "3. TOOL_CALL: Use this for requests needing real-time/live data, calculations, current time/date, search engines, or python execution (e.g. 'what is today's weather?', 'compute 1024 * 77', 'latest news about GPT').\n"
        "4. DIRECT_CHAT: Use this for general queries, greeting, chit-chat, or questions about the user's own memory (e.g. 'hello', 'who are you?', 'do you remember me?').\n\n"
        "Respond ONLY with one of: KNOWLEDGE_GRAPH, RAG, TOOL_CALL, DIRECT_CHAT."
    )
    
    try:
        response = await llm_manager.llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ])
        intent = response.content.strip().upper()
        if intent not in ["KNOWLEDGE_GRAPH", "RAG", "TOOL_CALL", "DIRECT_CHAT"]:
            intent = "DIRECT_CHAT"
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        intent = "DIRECT_CHAT"
        
    logger.info(f"[Node: classify_intent] Selected Intent: {intent}")
    return {"intent": intent}

async def retrieve_rag_context(state: AgentState) -> Dict[str, Any]:
    """Retrieves document text chunks using hybrid search."""
    query = state["user_query"]
    logger.info(f"[Node: retrieve_rag_context] Performing hybrid RAG search for: '{query}'")
    
    try:
        docs = await rag_pipeline.hybrid_search(query, k=3)
        contexts = [d.page_content for d in docs]
    except Exception as e:
        logger.error(f"RAG search node failed: {e}")
        contexts = []
        
    return {"retrieved_contexts": contexts}

async def retrieve_graph_context(state: AgentState) -> Dict[str, Any]:
    """Retrieves entities and relationships from the knowledge graph."""
    query = state["user_query"]
    logger.info(f"[Node: retrieve_graph_context] Fetching graph context for query: '{query}'")
    
    node_names = []
    node_details = []
    rel_details = []
    
    try:
        # Step 1: Find matching nodes in graph database
        matched_nodes = await db_query_graph_nodes(query)
        node_names = [n["name"] for n in matched_nodes[:3]]
        
        # Step 2: Fetch adjacent connections
        if node_names:
            neighborhood = await db_get_graph_neighborhood(node_names)
            node_details = [f"Node: {n['name']} ({n['label']}) - {n['description']}" for n in neighborhood["nodes"]]
            rel_details = [f"{r['source_id']} -- [{r['relation_type']}] --> {r['target_id']}" for r in neighborhood["edges"]]
    except Exception as e:
        logger.error(f"Graph context node failed: {e}")
        
    return {
        "graph_nodes": node_details,
        "graph_rels": rel_details
    }

async def execute_tools(state: AgentState) -> Dict[str, Any]:
    """Binds tools to LLM, lets it decide which tool to call, and runs it."""
    query = state["user_query"]
    logger.info(f"[Node: execute_tools] Deciding tool for query: '{query}'")
    
    tool_outputs = []
    try:
        # Bind our custom tools to the LLM if a valid OpenAI or Groq key is available
        if settings.has_valid_openai_key or settings.has_valid_groq_key:
            llm_with_tools = llm_manager.llm.bind_tools(ALL_TOOLS)
            response = await llm_with_tools.ainvoke([
                SystemMessage(content="You are an expert helper. Select and execute the most appropriate tool to answer the user query."),
                HumanMessage(content=query)
            ])
            
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    name = tool_call["name"]
                    args = tool_call["args"]
                    logger.info(f"[Node: execute_tools] Model selected tool {name} with args {args}")
                    
                    if name in TOOL_MAP:
                        # Invoke tool
                        result = TOOL_MAP[name].invoke(args)
                        tool_outputs.append(f"Tool {name} output: {result}")
                    else:
                        tool_outputs.append(f"Error: Tool {name} not found.")
            else:
                # Direct search fallback
                logger.info("[Node: execute_tools] No tool calls returned by model. Falling back to DuckDuckGo search.")
                search_res = TOOL_MAP["search_duckduckgo"].invoke(query)
                tool_outputs.append(f"Search results: {search_res}")
        else:
            # Fallback local tool choice (Regex/heuristics)
            if "time" in query.lower() or "date" in query.lower():
                res = TOOL_MAP["get_current_datetime"].invoke({})
            elif any(c in query for c in ["+", "*", "-", "/"]) or "calculate" in query.lower():
                # Extract equation
                res = TOOL_MAP["run_python_code"].invoke({"code": f"print({query})"})
            elif "arxiv" in query.lower() or "paper" in query.lower():
                res = TOOL_MAP["search_arxiv"].invoke({"query": query})
            else:
                res = TOOL_MAP["search_duckduckgo"].invoke({"query": query})
            tool_outputs.append(res)
            
    except Exception as e:
        logger.error(f"Tool execution node failed: {e}")
        tool_outputs.append(f"Tool failure: {str(e)}")
        
    return {"tool_outputs": tool_outputs}

async def generate_response(state: AgentState) -> Dict[str, Any]:
    """Assembles all context blocks and generates response using the LLM."""
    query = state["user_query"]
    profile = state["memory_profile"]
    relevant_mem = "\n".join(state["relevant_memories"])
    
    contexts = "\n\n".join(state["retrieved_contexts"])
    graph_nodes = "\n".join(state["graph_nodes"])
    graph_rels = "\n".join(state["graph_rels"])
    tool_outs = "\n\n".join(state["tool_outputs"])
    
    feedback = state.get("feedback", "")
    retry_count = state.get("retry_count", 0)
    
    logger.info(f"[Node: generate_response] Assembling prompt. Retry count: {retry_count}")
    
    system_prompt = (
        "You are an advanced Memory-Augmented Assistant. Synthesize all provided user profile data, "
        "recalled memory facts, vector context, relationship links, and tool outputs to formulate a highly personalized answer.\n\n"
        "Constraints:\n"
        "- Address the user based on their stored details (if name/preferences are known).\n"
        "- Cite source documents (e.g. [source_file.pdf] or [http://url]) if facts are retrieved from RAG.\n"
        "- Answer the query truthfully based on context. If details are not present, say so.\n\n"
        "--- USER DETAILS & PROFILE ---\n"
        f"{profile}\n\n"
        "--- SEMANTIC RECALLED FACTS ---\n"
        f"{relevant_mem}\n\n"
        "--- RETRIEVED DOCUMENT DATA (RAG) ---\n"
        f"{contexts if contexts else 'No RAG documents retrieved.'}\n\n"
        "--- KNOWLEDGE GRAPH DATA (NEO4J) ---\n"
        f"Nodes:\n{graph_nodes if graph_nodes else 'None'}\n"
        f"Edges:\n{graph_rels if graph_rels else 'None'}\n\n"
        "--- REAL-TIME API / TOOL OUTPUTS ---\n"
        f"{tool_outs if tool_outs else 'No tool executions.'}\n\n"
    )
    
    if feedback:
        system_prompt += f"\n[CORRECTION FEEDBACK FROM PREVIOUS RETRY]:\n{feedback}\nEnsure you fix this issue in your new response."
        
    try:
        response = await llm_manager.llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ])
        final_text = response.content.strip()
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        final_text = "I encountered an error formulating my response. Please check backend logs."
        
    return {
        "final_response": final_text,
        "retry_count": retry_count + 1
    }

async def evaluate_response(state: AgentState) -> Dict[str, Any]:
    """Judge LLM response for Faithfulness (hallucinations) and Query Relevance."""
    response = state["final_response"]
    contexts = state["retrieved_contexts"]
    query = state["user_query"]
    retry_count = state["retry_count"]
    
    logger.info("[Node: evaluate_response] Running LLM self-evaluation judge...")
    
    # If no RAG context retrieved, there is nothing to hallucinate against in RAG
    if not contexts:
        return {"evaluation_passed": True, "feedback": ""}
        
    judge_prompt = (
        "You are an LLM hallucination judge. Compare the assistant's answer with the retrieved context blocks.\n"
        "Determine if the assistant claims facts that are NOT supported by the retrieved contexts (hallucination).\n\n"
        "Retrieved Contexts:\n"
        f"{'|'.join(contexts)}\n\n"
        "Assistant Answer:\n"
        f"{response}\n\n"
        "Respond ONLY with a JSON object:\n"
        "{\n"
        "  \"passed\": true | false,\n"
        "  \"reason\": \"If failed, state exactly what claim is hallucinated or unsupported. Otherwise leave empty.\"\n"
        "}"
    )
    
    try:
        res = await llm_manager.llm.ainvoke([SystemMessage(content=judge_prompt)])
        raw = res.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\n", "", raw)
            raw = re.sub(r"\n```$", "", raw)
            raw = raw.strip()
            
        data = json.loads(raw)
        passed = bool(data.get("passed", True))
        reason = data.get("reason", "")
    except Exception as e:
        logger.warning(f"Self-evaluation node error: {e}. Defaulting to passed.")
        passed = True
        reason = ""
        
    # If failed, but we reached max retries (2), force pass to prevent infinite loop
    if not passed and retry_count >= 2:
        logger.warning("[Node: evaluate_response] Hallucination detected but retry limit reached. Proceeding to exit.")
        passed = True
        
    logger.info(f"[Node: evaluate_response] Passed evaluation: {passed}. Reason: '{reason}'")
    return {
        "evaluation_passed": passed,
        "feedback": reason if not passed else ""
    }


# ==========================================
# CONDITIONAL ROUTING EDGES
# ==========================================

def route_by_intent(state: AgentState) -> str:
    """Routes state execution based on detected intent."""
    intent = state["intent"]
    if intent == "RAG":
        return "rag_retrieval_node"
    elif intent == "KNOWLEDGE_GRAPH":
        return "knowledge_graph_node"
    elif intent == "TOOL_CALL":
        return "tool_execution_node"
    else:
        return "generate_response"

def route_evaluation(state: AgentState) -> str:
    """Decides whether to output response or loop back for regeneration."""
    if state["evaluation_passed"]:
        return END
    else:
        return "generate_response"


# ==========================================
# COMPILE WORKFLOW GRAPH
# ==========================================

workflow = StateGraph(AgentState)

# Register Nodes
workflow.add_node("retrieve_user_context", retrieve_user_context)
workflow.add_node("classify_intent", classify_intent)
workflow.add_node("rag_retrieval_node", retrieve_rag_context)
workflow.add_node("knowledge_graph_node", retrieve_graph_context)
workflow.add_node("tool_execution_node", execute_tools)
workflow.add_node("generate_response", generate_response)
workflow.add_node("evaluate_response", evaluate_response)

# Define Transitions
workflow.set_entry_point("retrieve_user_context")
workflow.add_edge("retrieve_user_context", "classify_intent")

# Conditional intent routing
workflow.add_conditional_edges(
    "classify_intent",
    route_by_intent,
    {
        "rag_retrieval_node": "rag_retrieval_node",
        "knowledge_graph_node": "knowledge_graph_node",
        "tool_execution_node": "tool_execution_node",
        "generate_response": "generate_response"
    }
)

# Connect RAG, Graph, and Tools nodes to generation node
workflow.add_edge("rag_retrieval_node", "generate_response")
workflow.add_edge("knowledge_graph_node", "generate_response")
workflow.add_edge("tool_execution_node", "generate_response")

# Loop response generation into evaluator
workflow.add_edge("generate_response", "evaluate_response")

# Conditional retry loop routing
workflow.add_conditional_edges(
    "evaluate_response",
    route_evaluation,
    {
        END: END,
        "generate_response": "generate_response"
    }
)

# Compile graph
compiled_agent = workflow.compile()
