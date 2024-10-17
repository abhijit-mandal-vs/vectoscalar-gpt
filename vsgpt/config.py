import os
from pathlib import Path
from datetime import datetime


class Config:
    class Path:
        ROOT_DIR = Path(__file__).parent.parent
        DATABASE_DIR = ROOT_DIR / "docs-db"
        DOCUMENTS_DIR = ROOT_DIR / "temp"
        IMAGES_DIR = ROOT_DIR / "images"
        
    class Database:
        DOCUMENTS_COLLECTION = "documents"
        
    class Model:
        EMBEDDINGS = "BAAI/bge-base-en-v1.5"
        RERANKER = "ms-marco-MiniLM-L-12-v2"
        LOCAL_LLM = "llama3.2:1b"
        REMOTE_LLM = "llama-3.1-70b-versatile"
        TEMPERATURE = 0.0
        MAX_TOKENS = 8000
        USE_LOCAL = False
        KEEP_ALIVE = 1
        
    class Retriever:
        USE_RERANKER = True
        USE_CHAIN_FILTER = False
        SHOW_SOURCES = False
        LAST_TRAINED = None
        FILES_INGESTED_COUNT = 0
        
        @classmethod
        def update_last_trained(cls, timestamp):
            cls.LAST_TRAINED = timestamp
        
        @classmethod
        def get_last_trained(cls):
            return cls.LAST_TRAINED
        
        @classmethod
        def update_files_ingested_count(cls, count):
            cls.FILES_INGESTED_COUNT = count

        @classmethod
        def get_files_ingested_count(cls):
            return cls.FILES_INGESTED_COUNT
        
    DEBUG = False
    CONVERSATION_MESSAGES_LIMIT = 100
    VERSION = "v1.o"