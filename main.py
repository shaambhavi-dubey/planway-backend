"""
SuperAgent Chat Backend - FastAPI application entry point.

Bootstraps services, registers routers, and starts the uvicorn server.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import settings
from app.routes import chat as chat_routes
from app.routes import composio as composio_routes
from app.routes import health as health_routes
from app.routes import rag as rag_routes
from app.services.chromadb_service import ChromaDBService
from app.services.composio_service import ComposioService
from app.services.conversation_service import ConversationService
from app.services.embedding_service import EmbeddingService
from app.services.gemini_embedding_service import GeminiEmbeddingService
from app.services.llm_service import LLMService
from app.services.pdf_service import PDFService
from app.services.superagent_service import SuperAgentService
from app.services.tool_executor import ToolExecutor

# -- Logging --

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# -- Service instances --

llm_service = LLMService()
conversation_service = ConversationService()
composio_service = ComposioService()
gemini_embedding_service = GeminiEmbeddingService()
embedding_service = EmbeddingService(gemini_service=gemini_embedding_service)
chromadb_service = ChromaDBService(gemini_service=gemini_embedding_service)
pdf_service = PDFService()
tool_executor = ToolExecutor(
    composio_service=composio_service,
    chromadb_service=chromadb_service,
)
superagent_service = SuperAgentService(
    llm_service=llm_service,
    conversation_service=conversation_service,
    composio_service=composio_service,
    tool_executor=tool_executor,
)


# -- Application lifecycle --


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle hook."""

    logger.info("Starting SuperAgent backend...")

    # Initialise active services
    await conversation_service.initialize()
    await llm_service.initialize()

    # Composio - initialises only when COMPOSIO_API_KEY is set
    await composio_service.initialize()

    # RAG pipeline - initialises only when relevant env vars are set
    await gemini_embedding_service.initialize()
    await embedding_service.initialize()
    await chromadb_service.initialize()
    await pdf_service.initialize()
    await tool_executor.initialize()

    # SuperAgent
    await superagent_service.initialize()

    # Wire services into routes
    chat_routes.configure(
        llm=llm_service,
        conversations=conversation_service,
        superagent=superagent_service,
    )
    composio_routes.configure(service=composio_service)
    rag_routes.configure(
        chromadb_service=chromadb_service,
        embedding_service=embedding_service,
        pdf_service=pdf_service,
    )

    logger.info("All services ready.")

    yield  # ← application runs here

    logger.info("Shutting down...")
    await superagent_service.shutdown()
    await tool_executor.shutdown()
    await pdf_service.shutdown()
    await chromadb_service.shutdown()
    await embedding_service.shutdown()
    await gemini_embedding_service.shutdown()
    await composio_service.shutdown()
    await llm_service.shutdown()
    await conversation_service.shutdown()


# -- FastAPI app --

app = FastAPI(
    title="Planway Chat Backend",
    description="LLM-powered chatting backend with conversation history and extensible tool execution.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS - dynamic origins from settings
origins = [
    "http://localhost:3000",
    "https://planway-app.vercel.app"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Register routers --

app.include_router(health_routes.router)
app.include_router(chat_routes.router)
app.include_router(composio_routes.router)
app.include_router(rag_routes.router)


# -- Standalone run --

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
