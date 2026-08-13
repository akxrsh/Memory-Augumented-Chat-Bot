import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from backend.database.connection import db_manager
from backend.schemas.api_schemas import UserRegister, MemoryItem, NodeSchema, RelationshipSchema
from backend.utils.logger import logger

# ==========================================
# USER CRUD
# ==========================================

async def db_create_user(user_id: str, username: str, password_hash: str, role: str = "user") -> Dict[str, Any]:
    """Creates a user in MongoDB if active, otherwise SQLite."""
    created_at = datetime.utcnow()
    user_doc = {
        "id": user_id,
        "username": username,
        "password_hash": password_hash,
        "role": role,
        "created_at": created_at
    }
    
    if not db_manager.use_fallback and db_manager.mongo_db is not None:
        try:
            await db_manager.mongo_db.users.insert_one(user_doc)
            return user_doc
        except Exception as e:
            logger.error(f"MongoDB write failed: {e}. Defaulting to SQLite.")
            
    # SQLite Fallback
    cursor = db_manager.sqlite_conn.cursor()
    cursor.execute(
        "INSERT INTO users (id, username, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, username, password_hash, role, created_at.isoformat())
    )
    db_manager.sqlite_conn.commit()
    return user_doc

async def db_get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Retrieves a user by username."""
    if not db_manager.use_fallback and db_manager.mongo_db is not None:
        try:
            return await db_manager.mongo_db.users.find_one({"username": username})
        except Exception as e:
            logger.error(f"MongoDB read failed: {e}. Defaulting to SQLite.")
            
    # SQLite Fallback
    cursor = db_manager.sqlite_conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    if row:
        return {
            "id": row["id"],
            "username": row["username"],
            "password_hash": row["password_hash"],
            "role": row["role"],
            "created_at": datetime.fromisoformat(row["created_at"]) if isinstance(row["created_at"], str) else row["created_at"]
        }
    return None

async def db_get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a user by user_id."""
    if not db_manager.use_fallback and db_manager.mongo_db is not None:
        try:
            return await db_manager.mongo_db.users.find_one({"id": user_id})
        except Exception as e:
            logger.error(f"MongoDB read failed: {e}. Defaulting to SQLite.")
            
    # SQLite Fallback
    cursor = db_manager.sqlite_conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    if row:
        return {
            "id": row["id"],
            "username": row["username"],
            "password_hash": row["password_hash"],
            "role": row["role"],
            "created_at": datetime.fromisoformat(row["created_at"]) if isinstance(row["created_at"], str) else row["created_at"]
        }
    return None


# ==========================================
# MEMORY CRUD
# ==========================================

async def db_add_memory(user_id: str, text: str, category: str, importance: int) -> Dict[str, Any]:
    """Adds a long-term memory node."""
    memory_id = str(uuid.uuid4())
    now = datetime.utcnow()
    memory_doc = {
        "id": memory_id,
        "user_id": user_id,
        "text": text,
        "category": category,
        "importance": importance,
        "created_at": now,
        "updated_at": now,
        "frequency": 1
    }
    
    if not db_manager.use_fallback and db_manager.mongo_db is not None:
        try:
            await db_manager.mongo_db.memories.insert_one(memory_doc)
            return memory_doc
        except Exception as e:
            logger.error(f"MongoDB write failed: {e}. Defaulting to SQLite.")
            
    # SQLite Fallback
    cursor = db_manager.sqlite_conn.cursor()
    cursor.execute(
        "INSERT INTO memories (id, user_id, text, category, importance, created_at, updated_at, frequency) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (memory_id, user_id, text, category, importance, now.isoformat(), now.isoformat(), 1)
    )
    db_manager.sqlite_conn.commit()
    return memory_doc

async def db_get_memories(user_id: str) -> List[Dict[str, Any]]:
    """Retrieves all memories for a specific user."""
    if not db_manager.use_fallback and db_manager.mongo_db is not None:
        try:
            cursor = db_manager.mongo_db.memories.find({"user_id": user_id})
            return await cursor.to_list(length=1000)
        except Exception as e:
            logger.error(f"MongoDB read failed: {e}. Defaulting to SQLite.")
            
    # SQLite Fallback
    cursor = db_manager.sqlite_conn.cursor()
    cursor.execute("SELECT * FROM memories WHERE user_id = ?", (user_id,))
    rows = cursor.fetchall()
    memories = []
    for r in rows:
        memories.append({
            "id": r["id"],
            "user_id": r["user_id"],
            "text": r["text"],
            "category": r["category"],
            "importance": r["importance"],
            "created_at": datetime.fromisoformat(r["created_at"]) if isinstance(r["created_at"], str) else r["created_at"],
            "updated_at": datetime.fromisoformat(r["updated_at"]) if isinstance(r["updated_at"], str) else r["updated_at"],
            "frequency": r["frequency"]
        })
    return memories

async def db_update_memory(memory_id: str, updates: Dict[str, Any]) -> bool:
    """Updates a memory document."""
    updates["updated_at"] = datetime.utcnow()
    
    if not db_manager.use_fallback and db_manager.mongo_db is not None:
        try:
            result = await db_manager.mongo_db.memories.update_one(
                {"id": memory_id},
                {"$set": updates}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"MongoDB update failed: {e}. Defaulting to SQLite.")
            
    # SQLite Fallback
    cursor = db_manager.sqlite_conn.cursor()
    set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
    values = list(updates.values())
    values.append(memory_id)
    
    cursor.execute(f"UPDATE memories SET {set_clause} WHERE id = ?", tuple(values))
    db_manager.sqlite_conn.commit()
    return cursor.rowcount > 0

async def db_delete_memory(memory_id: str) -> bool:
    """Deletes a memory document."""
    if not db_manager.use_fallback and db_manager.mongo_db is not None:
        try:
            result = await db_manager.mongo_db.memories.delete_one({"id": memory_id})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"MongoDB delete failed: {e}. Defaulting to SQLite.")
            
    # SQLite Fallback
    cursor = db_manager.sqlite_conn.cursor()
    cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    db_manager.sqlite_conn.commit()
    return cursor.rowcount > 0


# ==========================================
# CONVERSATION SESSION CRUD
# ==========================================

async def db_get_or_create_conversation(session_id: str, user_id: str) -> Dict[str, Any]:
    """Retrieves conversation history or initializes it."""
    if not db_manager.use_fallback and db_manager.mongo_db is not None:
        try:
            conv = await db_manager.mongo_db.conversations.find_one({"session_id": session_id})
            if conv:
                return conv
            
            new_conv = {
                "session_id": session_id,
                "user_id": user_id,
                "title": "New Chat",
                "messages": [],
                "summary": "",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            await db_manager.mongo_db.conversations.insert_one(new_conv)
            return new_conv
        except Exception as e:
            logger.error(f"MongoDB conversation fetch/create failed: {e}. Defaulting to SQLite.")
            
    # SQLite Fallback
    cursor = db_manager.sqlite_conn.cursor()
    cursor.execute("SELECT * FROM conversations WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    if row:
        return {
            "session_id": row["session_id"],
            "user_id": row["user_id"],
            "title": row["title"],
            "messages": json.loads(row["messages_json"]),
            "summary": row["summary"],
            "created_at": datetime.fromisoformat(row["created_at"]) if isinstance(row["created_at"], str) else row["created_at"],
            "updated_at": datetime.fromisoformat(row["updated_at"]) if isinstance(row["updated_at"], str) else row["updated_at"]
        }
        
    now = datetime.utcnow()
    new_conv = {
        "session_id": session_id,
        "user_id": user_id,
        "title": "New Chat",
        "messages": [],
        "summary": "",
        "created_at": now,
        "updated_at": now
    }
    cursor.execute(
        "INSERT INTO conversations (session_id, user_id, title, messages_json, summary, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (session_id, user_id, "New Chat", json.dumps([]), "", now.isoformat(), now.isoformat())
    )
    db_manager.sqlite_conn.commit()
    return new_conv

async def db_save_conversation(session_id: str, messages: List[Dict[str, Any]], summary: str = "", title: Optional[str] = None) -> bool:
    """Saves conversation message history."""
    now = datetime.utcnow()
    
    if not db_manager.use_fallback and db_manager.mongo_db is not None:
        try:
            update_doc = {
                "messages": messages,
                "updated_at": now
            }
            if summary:
                update_doc["summary"] = summary
            if title:
                update_doc["title"] = title
                
            result = await db_manager.mongo_db.conversations.update_one(
                {"session_id": session_id},
                {"$set": update_doc}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"MongoDB conversation save failed: {e}. Defaulting to SQLite.")
            
    # SQLite Fallback
    cursor = db_manager.sqlite_conn.cursor()
    if title and summary:
        cursor.execute(
            "UPDATE conversations SET messages_json = ?, summary = ?, title = ?, updated_at = ? WHERE session_id = ?",
            (json.dumps(messages), summary, title, now.isoformat(), session_id)
        )
    elif summary:
        cursor.execute(
            "UPDATE conversations SET messages_json = ?, summary = ?, updated_at = ? WHERE session_id = ?",
            (json.dumps(messages), summary, now.isoformat(), session_id)
        )
    elif title:
        cursor.execute(
            "UPDATE conversations SET messages_json = ?, title = ?, updated_at = ? WHERE session_id = ?",
            (json.dumps(messages), title, now.isoformat(), session_id)
        )
    else:
        cursor.execute(
            "UPDATE conversations SET messages_json = ?, updated_at = ? WHERE session_id = ?",
            (json.dumps(messages), now.isoformat(), session_id)
        )
    db_manager.sqlite_conn.commit()
    return cursor.rowcount > 0

async def db_get_all_user_conversations(user_id: str) -> List[Dict[str, Any]]:
    """Gets list of all chat sessions for a user."""
    if not db_manager.use_fallback and db_manager.mongo_db is not None:
        try:
            cursor = db_manager.mongo_db.conversations.find({"user_id": user_id})
            return await cursor.to_list(length=100)
        except Exception as e:
            logger.error(f"MongoDB query failed: {e}. Defaulting to SQLite.")
            
    # SQLite Fallback
    cursor = db_manager.sqlite_conn.cursor()
    cursor.execute("SELECT session_id, user_id, title, summary, created_at, updated_at FROM conversations WHERE user_id = ? ORDER BY updated_at DESC", (user_id,))
    rows = cursor.fetchall()
    return [dict(r) for r in rows]


# ==========================================
# KNOWLEDGE GRAPH DB OPERATORS
# ==========================================

async def db_add_graph_node(name: str, label: str, description: Optional[str] = None) -> NodeSchema:
    """Adds a graph node to Neo4j if active, otherwise SQLite fallback."""
    node_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, name))
    
    if not db_manager.use_fallback and db_manager.neo4j_driver is not None:
        try:
            async def _create_node(tx):
                query = (
                    "MERGE (n:Concept {name: $name}) "
                    "SET n.id = $id, n.label = $label, n.description = $description "
                    "RETURN n"
                )
                await tx.run(query, name=name, id=node_id, label=label, description=description)
            async with db_manager.neo4j_driver.session() as session:
                await session.execute_write(_create_node)
            return NodeSchema(id=node_id, name=name, label=label, description=description)
        except Exception as e:
            logger.error(f"Neo4j node creation failed: {e}. Defaulting to SQLite.")
            
    # SQLite Fallback
    cursor = db_manager.sqlite_conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO graph_nodes (id, name, label, description) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET label=excluded.label, description=excluded.description",
            (node_id, name, label, description)
        )
        db_manager.sqlite_conn.commit()
    except Exception as e:
        logger.error(f"SQLite graph node save failed: {e}")
        
    return NodeSchema(id=node_id, name=name, label=label, description=description)

async def db_add_graph_relationship(source_name: str, target_name: str, relation_type: str, weight: float = 1.0) -> RelationshipSchema:
    """Adds a graph relationship to Neo4j, otherwise SQLite fallback."""
    rel_id = str(uuid.uuid4())
    
    # Pre-create/get the nodes
    source_node = await db_add_graph_node(source_name, "Concept")
    target_node = await db_add_graph_node(target_name, "Concept")
    
    if not db_manager.use_fallback and db_manager.neo4j_driver is not None:
        try:
            async def _create_rel(tx):
                # Dynamically construct Cypher relations can be tricky, but merging nodes then creating relation is safe
                query = (
                    "MATCH (a:Concept {name: $source}), (b:Concept {name: $target}) "
                    "MERGE (a)-[r:RELATED {type: $rel_type}]->(b) "
                    "SET r.id = $id, r.weight = $weight "
                    "RETURN r"
                )
                await tx.run(query, source=source_name, target=target_name, rel_type=relation_type, id=rel_id, weight=weight)
            async with db_manager.neo4j_driver.session() as session:
                await session.execute_write(_create_rel)
            return RelationshipSchema(id=rel_id, source_id=source_node.id, target_id=target_node.id, relation_type=relation_type, weight=weight)
        except Exception as e:
            logger.error(f"Neo4j relationship creation failed: {e}. Defaulting to SQLite.")
            
    # SQLite Fallback
    cursor = db_manager.sqlite_conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO graph_relationships (id, source_id, target_id, relation_type, weight) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(source_id, target_id, relation_type) DO UPDATE SET weight=excluded.weight",
            (rel_id, source_node.id, target_node.id, relation_type, weight)
        )
        db_manager.sqlite_conn.commit()
    except Exception as e:
        logger.error(f"SQLite graph relationship save failed: {e}")
        
    return RelationshipSchema(id=rel_id, source_id=source_node.id, target_id=target_node.id, relation_type=relation_type, weight=weight)

async def db_query_graph_nodes(query_str: str) -> List[Dict[str, Any]]:
    """Simple semantic graph node name query/retrieval."""
    # Match query_str as substring or exact in node names
    if not db_manager.use_fallback and db_manager.neo4j_driver is not None:
        try:
            async def _query_nodes(tx):
                query = (
                    "MATCH (n:Concept) WHERE toLower(n.name) CONTAINS toLower($search) "
                    "RETURN n.id as id, n.name as name, n.label as label, n.description as description LIMIT 50"
                )
                result = await tx.run(query, search=query_str)
                return [record.data() for record in await result.all()]
            async with db_manager.neo4j_driver.session() as session:
                return await session.execute_read(_query_nodes)
        except Exception as e:
            logger.error(f"Neo4j graph query failed: {e}. Defaulting to SQLite.")
            
    # SQLite Fallback
    cursor = db_manager.sqlite_conn.cursor()
    cursor.execute(
        "SELECT id, name, label, description FROM graph_nodes WHERE name LIKE ? ORDER BY rowid DESC LIMIT 50",
        (f"%{query_str}%",)
    )
    rows = cursor.fetchall()
    return [dict(r) for r in rows]

async def db_get_graph_neighborhood(node_names: List[str]) -> Dict[str, Any]:
    """Retrieves nodes and edges adjacent to target node names."""
    nodes = []
    edges = []
    
    if not node_names:
        return {"nodes": nodes, "edges": edges}

    if not db_manager.use_fallback and db_manager.neo4j_driver is not None:
        try:
            async def _query_neighborhood(tx):
                query = (
                    "MATCH (n:Concept)-[r]->(m:Concept) "
                    "WHERE n.name IN $names OR m.name IN $names "
                    "RETURN n, r, m LIMIT 100"
                )
                result = await tx.run(query, names=node_names)
                records = await result.all()
                
                ret_nodes = {}
                ret_rels = []
                for rec in records:
                    n = rec["n"]
                    m = rec["m"]
                    r = rec["r"]
                    
                    ret_nodes[n["id"]] = {"id": n["id"], "name": n["name"], "label": n.get("label", "Concept"), "description": n.get("description", "")}
                    ret_nodes[m["id"]] = {"id": m["id"], "name": m["name"], "label": m.get("label", "Concept"), "description": m.get("description", "")}
                    
                    ret_rels.append({
                        "id": r.get("id", str(uuid.uuid4())),
                        "source_id": n["id"],
                        "target_id": m["id"],
                        "relation_type": r.get("type", "RELATED"),
                        "weight": r.get("weight", 1.0)
                    })
                return {"nodes": list(ret_nodes.values()), "edges": ret_rels}
                
            async with db_manager.neo4j_driver.session() as session:
                return await session.execute_read(_query_neighborhood)
        except Exception as e:
            logger.error(f"Neo4j neighborhood query failed: {e}. Defaulting to SQLite.")

    # SQLite Fallback
    cursor = db_manager.sqlite_conn.cursor()
    placeholders = ",".join(["?"] * len(node_names))
    
    # Query matching connections
    query = f"""
        SELECT 
            gn1.id AS s_id, gn1.name AS s_name, gn1.label AS s_label, gn1.description AS s_desc,
            gn2.id AS t_id, gn2.name AS t_name, gn2.label AS t_label, gn2.description AS t_desc,
            gr.id AS r_id, gr.relation_type, gr.weight
        FROM graph_relationships gr
        JOIN graph_nodes gn1 ON gr.source_id = gn1.id
        JOIN graph_nodes gn2 ON gr.target_id = gn2.id
        WHERE gn1.name IN ({placeholders}) OR gn2.name IN ({placeholders})
        LIMIT 100
    """
    
    cursor.execute(query, tuple(node_names + node_names))
    rows = cursor.fetchall()
    
    ret_nodes = {}
    ret_rels = []
    
    for r in rows:
        ret_nodes[r["s_id"]] = {"id": r["s_id"], "name": r["s_name"], "label": r["s_label"], "description": r["s_desc"]}
        ret_nodes[r["t_id"]] = {"id": r["t_id"], "name": r["t_name"], "label": r["t_label"], "description": r["t_desc"]}
        ret_rels.append({
            "id": r["r_id"],
            "source_id": r["s_id"],
            "target_id": r["t_id"],
            "relation_type": r["relation_type"],
            "weight": r["weight"]
        })
        
    if not ret_nodes:
        cursor.execute("SELECT id, name, label, description FROM graph_nodes ORDER BY rowid DESC LIMIT 50")
        node_rows = cursor.fetchall()
        for r in node_rows:
            ret_nodes[r["id"]] = {"id": r["id"], "name": r["name"], "label": r["label"], "description": r["description"]}

    return {"nodes": list(ret_nodes.values()), "edges": ret_rels}
