from fastapi import APIRouter, Depends, HTTPException, status
from backend.schemas.api_schemas import EvaluationRequest, EvaluationResponse
from backend.evaluation.evaluator import response_evaluator
from backend.services.auth import get_current_user
from backend.utils.logger import logger

router = APIRouter(prefix="/evaluate", tags=["Evaluation Framework"])

@router.post("", response_model=EvaluationResponse)
async def evaluate_turn(
    payload: EvaluationRequest,
    current_user: dict = Depends(get_current_user)
):
    """Triggers an offline evaluation on a custom query, response, and retrieved contexts."""
    logger.info(f"User {current_user['username']} is evaluating chatbot turn.")
    try:
        report = await response_evaluator.evaluate_response(
            query=payload.user_query,
            response=payload.llm_response,
            contexts=payload.contexts
        )
        
        # Format textual report summary
        report_text = (
            f"Faithfulness Score: {report['faithfulness']:.2f}\n"
            f"Answer Relevance: {report['answer_relevance']:.2f}\n"
            f"Context Recall: {report['context_recall']:.2f}\n"
            f"Latency: {report['latency']:.2f}s\n"
            f"Cost: ${report['cost']:.6f}"
        )
        
        return {
            "faithfulness": report["faithfulness"],
            "answer_relevance": report["answer_relevance"],
            "context_recall": report["context_recall"],
            "hallucination_rate": report["hallucination_rate"],
            "latency_seconds": report["latency"],
            "token_usage": {
                "prompt_tokens": report["prompt_tokens"],
                "completion_tokens": report["output_tokens"]
            },
            "cost": report["cost"],
            "report": report_text
        }
    except Exception as e:
        logger.error(f"Evaluation request failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/metrics")
async def get_dashboard_metrics(
    current_user: dict = Depends(get_current_user)
):
    """Retrieves aggregated statistics of all logged chat evaluations for the analytics dashboard."""
    try:
        metrics = await response_evaluator.get_metrics_summary()
        return metrics
    except Exception as e:
        logger.error(f"Failed to fetch metrics summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
