# LangChain Ecosystem and LangGraph Orchestration

LangChain is a popular open-source framework designed to simplify the creation of applications using Large Language Models (LLMs). The ecosystem consists of multiple packages:
1. **LangChain Core**: Base abstractions and interface definitions.
2. **LangChain Community**: Third-party integrations (databases, vector stores, API tools).
3. **LangGraph**: An extension of LangChain specifically designed to build stateful, multi-actor applications with loops.

## LangGraph Architecture
LangGraph represents agent architectures as cyclic graphs. Each node in the graph represents a step or LLM execution, and edges represent transitions between states.
A StateGraph is initialized with a defined schema, representing the shared state of the graph. Nodes update this state, and conditional edges decide the routing.

## Knowledge Graph reasoning
Using a Knowledge Graph (like Neo4j) allows agents to reason over complex relationships. For example, query path:
- RAG retrieves document chunks.
- Knowledge Graph queries related concepts and technology families.
- Hybrid systems combine vector distance matching with graph relationships.
