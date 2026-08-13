print("--- [FastAPI Startup] main.py entry point reached ---", flush=True)
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from backend.config.settings import settings
from backend.database.connection import db_manager
from backend.utils.logger import logger

# Import API Routers
from backend.api.v1.auth import router as auth_router
from backend.api.v1.chat import router as chat_router
from backend.api.v1.documents import router as documents_router
from backend.api.v1.evaluate import router as evaluate_router
from backend.api.v1.graph import router as graph_router
from backend.api.v1.memory import router as memory_router
from backend.api.v1.tools import router as tools_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Operations
    logger.info("Starting up FastAPI application...")
    try:
        await db_manager.initialize()
        logger.info("Database managers initialized successfully.")
    except Exception as e:
        logger.error(f"Error during database initialization: {e}")
    
    yield
    
    # Shutdown Operations
    logger.info("Shutting down FastAPI application...")
    await db_manager.close()
    logger.info("Database connections shut down successfully.")

app = FastAPI(
    title=settings.APP_NAME,
    description="An enterprise-grade Memory-Augmented Chatbot System with Hybrid RAG, Knowledge Graph and LangGraph agents.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware Configurations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handling middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except Exception as e:
        logger.exception("Unhandled Server Exception")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal Server Error. Please see server logs for details."}
        )

# Include API Routers
app.include_router(auth_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")
app.include_router(evaluate_router, prefix="/api/v1")
app.include_router(graph_router, prefix="/api/v1")
app.include_router(memory_router, prefix="/api/v1")
app.include_router(tools_router, prefix="/api/v1")

# Mount Static Files for Frontend
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint to verify backend status."""
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "environment": settings.ENV,
        "database_fallback_mode": db_manager.use_fallback,
        "mongodb_connected": db_manager.mongo_db is not None,
        "neo4j_connected": db_manager.neo4j_driver is not None,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=False)
