import json
import re
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_core.messages import SystemMessage
from backend.services.llm import llm_manager
from backend.database.connection import db_manager
from backend.utils.logger import logger

class ResponseEvaluator:
    """Calculates Faithfulness, Answer Relevance, and Context Recall, logging reports to SQLite/MongoDB."""

    def __init__(self):
        self.llm = llm_manager.llm

    async def evaluate_response(
        self,
        query: str,
        response: str,
        contexts: List[str]
    ) -> Dict[str, Any]:
        """Runs LLM judges to score Faithfulness, Answer Relevance, and Context Recall."""
        logger.info("Executing evaluation pipeline...")
        
        start_time = time.time()
        
        # 1. Calculate Faithfulness (Groundedness)
        faithfulness = 1.0
        faithfulness_reason = "No contexts retrieved."
        
        if contexts:
            faithfulness_prompt = (
                "You are an AI faithfulness evaluator. Analyze the context and response below and verify if ALL claims "
                "in the response are fully supported by the contexts.\n\n"
                "Contexts:\n"
                f"{' | '.join(contexts)}\n\n"
                "Response:\n"
                f"{response}\n\n"
                "Respond ONLY with a JSON object:\n"
                "{\n"
                "  \"score\": 0.0 to 1.0,\n"
                "  \"reason\": \"Brief explanation of the score.\"\n"
                "}"
            )
            
            try:
                res = await self.llm.ainvoke([SystemMessage(content=faithfulness_prompt)])
                raw = res.content.strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```(?:json)?\n", "", raw)
                    raw = re.sub(r"\n```$", "", raw)
                    raw = raw.strip()
                data = json.loads(raw)
                faithfulness = float(data.get("score", 1.0))
                faithfulness_reason = data.get("reason", "")
            except Exception as e:
                logger.error(f"Faithfulness evaluation failed: {e}")
                faithfulness = 0.5
                faithfulness_reason = f"Evaluation failed: {str(e)}"

        # 2. Calculate Answer Relevance
        relevance = 1.0
        relevance_reason = ""
        relevance_prompt = (
            "You are an AI answer relevance evaluator. Analyze the user query and the assistant's response.\n"
            "Evaluate if the response directly and clearly addresses the query without fluff.\n\n"
            "Query: {query}\n"
            "Response: {response}\n\n"
            "Respond ONLY with a JSON object:\n"
            "{{\n"
            "  \"score\": 0.0 to 1.0,\n"
            "  \"reason\": \"Brief explanation of the score.\"\n"
            "}}"
        ).format(query=query, response=response)
        
        try:
            res = await self.llm.ainvoke([SystemMessage(content=relevance_prompt)])
            raw = res.content.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\n", "", raw)
                raw = re.sub(r"\n```$", "", raw)
                raw = raw.strip()
            data = json.loads(raw)
            relevance = float(data.get("score", 1.0))
            relevance_reason = data.get("reason", "")
        except Exception as e:
            logger.error(f"Relevance evaluation failed: {e}")
            relevance = 0.5

        # 3. Calculate Context Recall
        recall = 1.0
        recall_reason = ""
        if contexts:
            recall_prompt = (
                "You are an AI context recall evaluator. Determine if the retrieved context contains all necessary details "
                "to answer the user's query.\n\n"
                "Query: {query}\n"
                "Contexts:\n"
                f"{' | '.join(contexts)}\n\n"
                "Respond ONLY with a JSON object:\n"
                "{{\n"
                "  \"score\": 0.0 to 1.0,\n"
                "  \"reason\": \"Brief explanation of the score.\"\n"
                "}}"
            ).format(query=query)
            
            try:
                res = await self.llm.ainvoke([SystemMessage(content=recall_prompt)])
                raw = res.content.strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```(?:json)?\n", "", raw)
                    raw = re.sub(r"\n```$", "", raw)
                    raw = raw.strip()
                data = json.loads(raw)
                recall = float(data.get("score", 1.0))
                recall_reason = data.get("reason", "")
            except Exception as e:
                logger.error(f"Context recall evaluation failed: {e}")
                recall = 0.5
        else:
            recall = 0.0
            recall_reason = "No context retrieved."

        latency = time.time() - start_time
        
        # Calculate mock token cost (0.015 / 1k input tokens, 0.06 / 1k output tokens)
        prompt_tokens = len(query.split()) + sum(len(c.split()) for c in contexts) + 300
        output_tokens = len(response.split())
        cost = (prompt_tokens * 0.0000015) + (output_tokens * 0.000002)

        eval_doc = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "response": response,
            "faithfulness": faithfulness,
            "answer_relevance": relevance,
            "context_recall": recall,
            "hallucination_rate": 1.0 - faithfulness,
            "latency": latency,
            "cost": cost,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "faithfulness_reason": faithfulness_reason,
            "relevance_reason": relevance_reason,
            "recall_reason": recall_reason
        }

        # Save eval to db
        await self._log_evaluation(eval_doc)
        
        return eval_doc

    async def _log_evaluation(self, doc: Dict[str, Any]):
        """Saves evaluation records into MongoDB or SQLite fallback."""
        if not db_manager.use_fallback and db_manager.mongo_db is not None:
            try:
                await db_manager.mongo_db.evaluations.insert_one(doc)
                return
            except Exception as e:
                logger.error(f"Failed to log evaluation to MongoDB: {e}")
                
        # SQLite Fallback
        cursor = db_manager.sqlite_conn.cursor()
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    query TEXT,
                    response TEXT,
                    faithfulness REAL,
                    answer_relevance REAL,
                    context_recall REAL,
                    hallucination_rate REAL,
                    latency REAL,
                    cost REAL,
                    prompt_tokens INTEGER,
                    output_tokens INTEGER,
                    faithfulness_reason TEXT,
                    relevance_reason TEXT,
                    recall_reason TEXT
                )
            """)
            cursor.execute("""
                INSERT INTO evaluations (
                    timestamp, query, response, faithfulness, answer_relevance, context_recall, 
                    hallucination_rate, latency, cost, prompt_tokens, output_tokens, 
                    faithfulness_reason, relevance_reason, recall_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc["timestamp"], doc["query"], doc["response"], doc["faithfulness"], 
                doc["answer_relevance"], doc["context_recall"], doc["hallucination_rate"], 
                doc["latency"], doc["cost"], doc["prompt_tokens"], doc["output_tokens"],
                doc["faithfulness_reason"], doc["relevance_reason"], doc["recall_reason"]
            ))
            db_manager.sqlite_conn.commit()
        except Exception as e:
            logger.error(f"SQLite evaluation logging failed: {e}")

    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Retrieves aggregated scores and stats for the evaluation dashboard."""
        if not db_manager.use_fallback and db_manager.mongo_db is not None:
            try:
                # MongoDB aggregation
                pipeline = [
                    {
                        "$group": {
                            "_id": None,
                            "avg_faithfulness": {"$avg": "$faithfulness"},
                            "avg_relevance": {"$avg": "$answer_relevance"},
                            "avg_recall": {"$avg": "$context_recall"},
                            "avg_latency": {"$avg": "$latency"},
                            "total_cost": {"$sum": "$cost"},
                            "count": {"$sum": 1}
                        }
                    }
                ]
                cursor = db_manager.mongo_db.evaluations.aggregate(pipeline)
                results = await cursor.to_list(length=1)
                if results:
                    r = results[0]
                    return {
                        "average_faithfulness": r["avg_faithfulness"],
                        "average_relevance": r["avg_relevance"],
                        "average_recall": r["avg_recall"],
                        "average_latency": r["avg_latency"],
                        "total_cost": r["total_cost"],
                        "total_evaluations": r["count"]
                    }
            except Exception as e:
                logger.error(f"MongoDB metrics aggregation failed: {e}")

        # SQLite Fallback
        cursor = db_manager.sqlite_conn.cursor()
        try:
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='evaluations'")
            if not cursor.fetchone():
                return {
                    "average_faithfulness": 1.0, "average_relevance": 1.0, "average_recall": 1.0,
                    "average_latency": 0.0, "total_cost": 0.0, "total_evaluations": 0
                }
                
            cursor.execute("""
                SELECT 
                    AVG(faithfulness) as avg_faith, 
                    AVG(answer_relevance) as avg_rel, 
                    AVG(context_recall) as avg_rec,
                    AVG(latency) as avg_lat,
                    SUM(cost) as tot_cost,
                    COUNT(*) as cnt
                FROM evaluations
            """)
            row = cursor.fetchone()
            if row and row["cnt"] > 0:
                return {
                    "average_faithfulness": row["avg_faith"] or 0.0,
                    "average_relevance": row["avg_rel"] or 0.0,
                    "average_recall": row["avg_rec"] or 0.0,
                    "average_latency": row["avg_lat"] or 0.0,
                    "total_cost": row["tot_cost"] or 0.0,
                    "total_evaluations": row["cnt"]
                }
        except Exception as e:
            logger.error(f"SQLite metrics aggregation failed: {e}")
            
        return {
            "average_faithfulness": 1.0,
            "average_relevance": 1.0,
            "average_recall": 1.0,
            "average_latency": 0.0,
            "total_cost": 0.0,
            "total_evaluations": 0
        }

response_evaluator = ResponseEvaluator()
