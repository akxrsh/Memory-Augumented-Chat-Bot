import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Server Configuration
    APP_NAME: str = "Memory-Augmented Chatbot System"
    ENV: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Security & Auth
    JWT_SECRET: str = "supersecretjwtkeyforlocaldevelopmentonlychangeinprod!"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 1 day

    # LLM Settings
    OPENAI_API_KEY: Optional[str] = None
    LLM_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.1-8b-instant"

    # MongoDB Settings
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "chatbot_memory"

    # Neo4j Settings
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    # Vector DB Settings
    VECTOR_DB_DIR: str = "data/vector_db"
    CHROMA_DB_DIR: str = "data/chroma_db"

    # SQLite Local Fallback settings (if DBs are offline)
    USE_LOCAL_FALLBACK: bool = True
    LOCAL_DB_PATH: str = "data/local_fallback.db"

    # Rate Limiting
    RATE_LIMIT_CALLS: int = 100
    RATE_LIMIT_PERIOD_SECONDS: int = 60

    @property
    def has_valid_openai_key(self) -> bool:
        k = self.OPENAI_API_KEY
        if not k:
            return False
        k = k.strip()
        return bool(k and not k.startswith("your_") and "here" not in k)

    @property
    def has_valid_groq_key(self) -> bool:
        k = self.GROQ_API_KEY
        if not k:
            return False
        k = k.strip()
        return bool(k and not k.startswith("your_") and "here" not in k)

settings = Settings()

