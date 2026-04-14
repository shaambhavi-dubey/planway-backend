"""
Application settings loaded from environment variables / .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the chat backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- LLM --
    LLM_MODEL: str = "gemini/gemini-2.0-flash"
    LLM_API_KEY: str = ""
    LLM_TEMPERATURE: float = 0.6

    # -- Composio --
    COMPOSIO_API_KEY: str = ""
    COMPOSIO_ORG_KEY: str = ""
    COMPOSIO_BASE_URL: str = "https://backend.composio.dev/api/v2"

    # -- Gemini Embedding --
    GEMINI_EMBED_API_KEY: str = ""
    GEMINI_EMBED_MODEL: str = "gemini-embedding-001"
    GEMINI_EMBED_DIMENSION: int = 768

    # -- ChromaDB Cloud --
    CHROMADB_API_KEY: str = ""
    CHROMADB_TENANT: str = ""
    CHROMADB_DATABASE: str = ""
    CHROMADB_COLLECTION_NAME: str = "superagent_rag"

    # -- RAG --
    RAG_ENABLED: bool = False
    RAG_TOP_K: int = 5

    # FIX: Was 1.5 — but ChromaDB cosine distance range is [0, 2].
    # With hnsw:space=cosine:
    #   0.0  = identical vectors
    #   1.0  = orthogonal (unrelated)
    #   2.0  = opposite
    # A threshold of 1.5 means "keep anything less than 1.5" which is almost
    # everything including irrelevant results. For meaningful RAG retrieval,
    # 0.7 keeps only genuinely similar chunks.
    # Upload ingestion is unaffected by this — it only filters search results.
    RAG_SIMILARITY_THRESHOLD: float = 0.7

    # -- PDF Upload --
    PDF_CHUNK_SIZE: int = 1000
    PDF_CHUNK_OVERLAP: int = 200
    PDF_MAX_FILE_SIZE_MB: int = 50

    # -- Server --
    HOST: str = "0.0.0.0"
    PORT: int = 5050
    DEBUG: bool = False
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "https://planway-frontend.vercel.app"]


# Singleton instance
settings = Settings()