import os
import sqlite3
import asyncio
from typing import Any, Dict, List, Optional
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient
from neo4j import AsyncGraphDatabase, AsyncDriver
from backend.config.settings import settings
from backend.utils.logger import logger

class DatabaseManager:
    """Manages connections to MongoDB, Neo4j, and local SQLite fallback."""
    
    def __init__(self):
        self.mongo_client: Optional[AsyncIOMotorClient] = None
        self.mongo_db = None
        self.neo4j_driver: Optional[AsyncDriver] = None
        self.sqlite_conn: Optional[sqlite3.Connection] = None
        
        # Determine fallback mode
        self.use_fallback = settings.USE_LOCAL_FALLBACK

    def init_sqlite(self):
        """Initializes local SQLite database for fallback and auth schema."""
        db_path = Path(settings.LOCAL_DB_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing SQLite fallback database at: {db_path}")
        self.sqlite_conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.sqlite_conn.row_factory = sqlite3.Row
        
        cursor = self.sqlite_conn.cursor()
        
        # Create users table (shared for security/auth)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create fallback memories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                text TEXT NOT NULL,
                category TEXT,
                importance INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                frequency INTEGER DEFAULT 1
            )
        """)
        
        # Create fallback conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT,
                messages_json TEXT NOT NULL,
                summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create fallback graph nodes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS graph_nodes (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                label TEXT NOT NULL,
                description TEXT
            )
        """)
        
        # Create fallback graph relationships table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS graph_relationships (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                FOREIGN KEY(source_id) REFERENCES graph_nodes(id) ON DELETE CASCADE,
                FOREIGN KEY(target_id) REFERENCES graph_nodes(id) ON DELETE CASCADE,
                UNIQUE(source_id, target_id, relation_type)
            )
        """)
        
        self.sqlite_conn.commit()

    async def connect_mongo(self) -> bool:
        """Tries to connect to MongoDB, sets fallback if unable."""
        if self.use_fallback:
            logger.info("MongoDB local fallback mode is forced via settings.")
            return False

        try:
            logger.info("Connecting to MongoDB...")
            self.mongo_client = AsyncIOMotorClient(
                settings.MONGODB_URI,
                serverSelectionTimeoutMS=2000
            )
            # Trigger server_info to test connection status
            await self.mongo_client.server_info()
            self.mongo_db = self.mongo_client[settings.MONGODB_DB_NAME]
            logger.info("Successfully connected to MongoDB.")
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to MongoDB: {e}. Falling back to SQLite.")
            self.mongo_client = None
            self.mongo_db = None
            return False

    async def connect_neo4j(self) -> bool:
        """Tries to connect to Neo4j database, sets fallback if unable."""
        if self.use_fallback:
            logger.info("Neo4j local fallback mode is forced via settings.")
            return False

        try:
            logger.info("Connecting to Neo4j...")
            self.neo4j_driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            # Verify connectivity
            await self.neo4j_driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j.")
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to Neo4j: {e}. Falling back to SQLite.")
            self.neo4j_driver = None
            return False

    async def initialize(self):
        """Initializes all database connection wrappers."""
        # SQLite schema always initialized for local auth & mock support
        self.init_sqlite()

        mongo_ok = await self.connect_mongo()
        neo4j_ok = await self.connect_neo4j()
        
        if not mongo_ok or not neo4j_ok:
            logger.info("One or more external databases are offline. Running in Hybrid Fallback mode (SQLite).")
            self.use_fallback = True
        else:
            self.use_fallback = False

    async def close(self):
        """Closes all active database connections."""
        if self.mongo_client:
            self.mongo_client.close()
            logger.info("MongoDB connection closed.")
        if self.neo4j_driver:
            await self.neo4j_driver.close()
            logger.info("Neo4j connection closed.")
        if self.sqlite_conn:
            self.sqlite_conn.close()
            logger.info("SQLite fallback connection closed.")

db_manager = DatabaseManager()
