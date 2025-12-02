"""Configuration settings for the RAG pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_INPUT_DIR = Path(os.getenv("DATA_INPUT_DIR", PROJECT_ROOT / "data"))
DATA_OUTPUT_DIR = Path(os.getenv("DATA_OUTPUT_DIR", PROJECT_ROOT / "data"))


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    llm_model: str = Field(default="gemini-2.5-flash-lite", alias="LLM_MODEL")
    embedding_model: str = Field(
        default="bkai-foundation-models/vietnamese-bi-encoder",
        alias="EMBEDDING_MODEL",
    )
    qdrant_collection: str = Field(
        default="vnpt_knowledge_base",
        alias="QDRANT_COLLECTION",
    )
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_retrieval: int = 3

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

