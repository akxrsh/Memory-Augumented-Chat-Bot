from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Any
from backend.tools.custom_tools import search_duckduckgo, run_python_code
from backend.services.auth import get_current_user
from backend.utils.logger import logger

router = APIRouter(prefix="/tools", tags=["Dynamic Tools"])

class SearchQuery(BaseModel):
    query: str

class PythonCodeRequest(BaseModel):
    code: str

@router.post("/search")
async def search_web(
    payload: SearchQuery,
    current_user: dict = Depends(get_current_user)
):
    """Executes a web search query on DuckDuckGo."""
    try:
        # Run search tool (wrapped inside LangChain Tool, callable with string argument)
        results = search_duckduckgo.invoke(payload.query)
        return {"status": "success", "results": results}
    except Exception as e:
        logger.error(f"Manual search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/execute")
async def execute_python(
    payload: PythonCodeRequest,
    current_user: dict = Depends(get_current_user)
):
    """Executes Python code in a safe sandbox."""
    try:
        output = run_python_code.invoke(payload.code)
        return {"status": "success", "output": output}
    except Exception as e:
        logger.error(f"Manual Python code execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
