"""
ChromaDB Cloud service for document storage and retrieval.

Manages a single collection with dense (Gemini) embeddings.
Provides add, search, and delete.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import chromadb

from app.config.settings import settings
from app.services.embedding_service import EmbeddingServiceChromaDBAdapter
from app.services.gemini_embedding_service import GeminiEmbeddingService
from app.services.base.base_service import BaseService

logger = logging.getLogger(__name__)


class ChromaDBService(BaseService):
    """Singleton-style service wrapping a ChromaDB collection."""

    def __init__(self, gemini_service: GeminiEmbeddingService) -> None:
        self._gemini = gemini_service
        self._ef: Optional[EmbeddingServiceChromaDBAdapter] = None
        self._client: Any = None
        self._collection: Any = None

    # -- Lifecycle --

    async def initialize(self) -> None:
        if not self._gemini.is_initialized:
            logger.error(
                "ChromaDBService: GeminiEmbeddingService is NOT initialized. "
                "Check that GEMINI_EMBED_API_KEY is set in your .env file."
            )
            return

        try:
            self._ef = EmbeddingServiceChromaDBAdapter(self._gemini)

            # Determine which backend to use and log it clearly
            use_cloud = all([
                settings.CHROMADB_API_KEY,
                settings.CHROMADB_TENANT,
                settings.CHROMADB_DATABASE,
            ])

            if use_cloud:
                logger.info(
                    "Connecting to ChromaDB Cloud (tenant=%s, database=%s)...",
                    settings.CHROMADB_TENANT,
                    settings.CHROMADB_DATABASE,
                )
                self._client = chromadb.CloudClient(
                    api_key=settings.CHROMADB_API_KEY,
                    tenant=settings.CHROMADB_TENANT,
                    database=settings.CHROMADB_DATABASE,
                )
            else:
                # Log WHICH variable is missing so the user knows exactly what to fix
                missing = []
                if not settings.CHROMADB_API_KEY:
                    missing.append("CHROMADB_API_KEY")
                if not settings.CHROMADB_TENANT:
                    missing.append("CHROMADB_TENANT")
                if not settings.CHROMADB_DATABASE:
                    missing.append("CHROMADB_DATABASE")
                logger.warning(
                    "ChromaDB Cloud credentials incomplete (%s not set). "
                    "Falling back to LOCAL persistent ChromaDB at ./chroma_db. "
                    "Documents will be stored locally, not in the cloud.",
                    ", ".join(missing),
                )
                self._client = chromadb.PersistentClient(path="./chroma_db")

            collection_name = settings.CHROMADB_COLLECTION_NAME

            # Do NOT pass embedding_function — we supply pre-computed embeddings
            # directly in add() calls. Passing an embedding_function here causes
            # ChromaDB Cloud to try to invoke it server-side and fail.
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            count = await asyncio.to_thread(self._collection.count)
            logger.info(
                "ChromaDB collection '%s' ready — %d documents already stored, "
                "embedding_dim=%d",
                collection_name,
                count,
                self._ef.vector_dimension,
            )

            await super().initialize()

        except Exception as exc:
            logger.error(
                "ChromaDBService initialization failed: %s\n"
                "  CHROMADB_TENANT=%s\n"
                "  CHROMADB_DATABASE=%s\n"
                "  CHROMADB_API_KEY set=%s",
                exc,
                settings.CHROMADB_TENANT or "(empty)",
                settings.CHROMADB_DATABASE or "(empty)",
                bool(settings.CHROMADB_API_KEY),
                exc_info=True,
            )
            raise

    async def health_check(self) -> bool:
        if not self.is_initialized or self._collection is None:
            return False
        try:
            count = await asyncio.to_thread(self._collection.count)
            return count >= 0
        except Exception as exc:
            logger.error("ChromaDB health-check failed: %s", exc)
            return False

    async def shutdown(self) -> None:
        self._client = None
        self._collection = None
        self._ef = None
        await super().shutdown()

    # -- Document ingestion --

    async def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Add documents with pre-computed Gemini embeddings.

        Each failure stage now logs a distinct, actionable error message
        so you know exactly which step is broken.
        """
        self._ensure_initialized()

        metrics: Dict[str, Any] = {
            "success": False,
            "total_chunks": len(documents),
            "total_latency_ms": 0.0,
        }
        t0 = time.time()

        # -- Input validation --
        if not documents:
            logger.error("add_documents: received empty documents list")
            return metrics

        if len(documents) != len(metadatas):
            logger.error(
                "add_documents: length mismatch — %d documents vs %d metadatas",
                len(documents), len(metadatas),
            )
            return metrics

        if ids is None:
            ids = [f"chunk_{uuid.uuid4().hex}" for _ in range(len(documents))]

        logger.info(
            "add_documents: starting ingestion of %d chunks into collection '%s'",
            len(documents),
            settings.CHROMADB_COLLECTION_NAME,
        )

        # -- Sanitise metadata (ChromaDB only accepts str/int/float/bool) --
        processed_meta = []
        for md in metadatas:
            clean: Dict[str, Any] = {}
            for k, v in md.items():
                if isinstance(v, list):
                    clean[k] = ",".join(str(x) for x in v)
                elif v is None:
                    clean[k] = ""
                else:
                    clean[k] = v
            processed_meta.append(clean)

        assert self._ef is not None

        # -- Step 1: Generate embeddings --
        logger.info("Step 1/2: Generating Gemini embeddings for %d chunks...", len(documents))
        try:
            embeddings = await asyncio.to_thread(
                self._ef.get_embeddings, documents, "RETRIEVAL_DOCUMENT"
            )
        except Exception as exc:
            logger.error(
                "Step 1/2 FAILED — Gemini embedding generation raised an exception: %s\n"
                "  Check GEMINI_EMBED_API_KEY and GEMINI_EMBED_MODEL in your .env",
                exc,
                exc_info=True,
            )
            return metrics

        if not embeddings:
            logger.error(
                "Step 1/2 FAILED — Gemini returned empty embeddings list. "
                "Check GEMINI_EMBED_API_KEY, quota, and model name '%s'.",
                settings.GEMINI_EMBED_MODEL,
            )
            return metrics

        if len(embeddings) != len(documents):
            logger.error(
                "Step 1/2 FAILED — embedding count mismatch: got %d, expected %d",
                len(embeddings), len(documents),
            )
            return metrics

        # Filter chunks where embedding failed (embed_batch returns [] for failures)
        valid = [
            (ids[i], documents[i], embeddings[i], processed_meta[i])
            for i in range(len(documents))
            if embeddings[i] and len(embeddings[i]) > 0
        ]

        if not valid:
            logger.error(
                "Step 1/2 FAILED — ALL %d embeddings are empty vectors. "
                "This almost always means your GEMINI_EMBED_API_KEY is invalid "
                "or you have exceeded your Gemini API quota.",
                len(documents),
            )
            return metrics

        skipped = len(documents) - len(valid)
        if skipped:
            logger.warning(
                "Step 1/2 partial — %d/%d chunks had failed embeddings and will be skipped",
                skipped, len(documents),
            )

        logger.info(
            "Step 1/2 complete — %d/%d embeddings generated successfully (dim=%d)",
            len(valid), len(documents), len(valid[0][2]),
        )

        valid_ids   = [x[0] for x in valid]
        valid_docs  = [x[1] for x in valid]
        valid_embs  = [x[2] for x in valid]
        valid_metas = [x[3] for x in valid]

        # -- Step 2: Store in ChromaDB --
        logger.info(
            "Step 2/2: Storing %d chunks in ChromaDB collection '%s'...",
            len(valid_ids), settings.CHROMADB_COLLECTION_NAME,
        )

        def _add() -> None:
            self._collection.add(
                ids=valid_ids,
                documents=valid_docs,
                embeddings=valid_embs,
                metadatas=valid_metas,
            )

        try:
            await asyncio.to_thread(_add)
        except Exception as exc:
            logger.error(
                "Step 2/2 FAILED — ChromaDB add() raised: %s\n"
                "  collection=%s\n"
                "  chunks=%d\n"
                "  embedding_dim=%d\n"
                "  ChromaDB Cloud: tenant=%s, database=%s, api_key_set=%s\n"
                "  Possible causes:\n"
                "    - Duplicate chunk IDs (re-uploading same PDF)\n"
                "    - Embedding dimension mismatch with existing collection\n"
                "    - ChromaDB Cloud auth failure (wrong tenant/database/api_key)\n"
                "    - Network timeout",
                exc,
                settings.CHROMADB_COLLECTION_NAME,
                len(valid_ids),
                len(valid_embs[0]) if valid_embs else 0,
                settings.CHROMADB_TENANT or "(empty)",
                settings.CHROMADB_DATABASE or "(empty)",
                bool(settings.CHROMADB_API_KEY),
                exc_info=True,
            )
            return metrics

        latency = (time.time() - t0) * 1000
        metrics.update(
            success=True,
            total_chunks=len(valid_ids),
            total_latency_ms=round(latency, 2),
        )
        logger.info(
            "Step 2/2 complete — stored %d/%d chunks in %.2f ms",
            len(valid_ids), len(documents), latency,
        )
        return metrics

    # -- Search --

    async def search(
        self,
        query_text: str,
        n_results: int = 10,
        similarity_threshold: float = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Dense vector search using a Gemini query embedding.

        similarity_threshold: cosine distance cutoff — results with distance
        >= threshold are excluded. Lower = stricter. Default from settings.
        """
        self._ensure_initialized()
        assert self._ef is not None

        # Use settings default if not overridden
        if similarity_threshold is None:
            similarity_threshold = settings.RAG_SIMILARITY_THRESHOLD

        t0 = time.time()

        # Generate query embedding
        try:
            query_emb = await asyncio.to_thread(
                self._ef.get_embedding, query_text, "RETRIEVAL_QUERY"
            )
        except Exception as exc:
            logger.error("search: query embedding failed: %s", exc)
            return []

        if not query_emb:
            logger.error(
                "search: query embedding returned empty list — "
                "check Gemini service is healthy"
            )
            return []

        # Guard against querying an empty collection (ChromaDB raises on count=0)
        try:
            count = await asyncio.to_thread(self._collection.count)
        except Exception as exc:
            logger.error("search: could not get collection count: %s", exc)
            return []

        if count == 0:
            logger.warning("search: collection is empty, no results possible")
            return []

        safe_n = min(n_results * 3, count)

        query_params: Dict[str, Any] = {
            "query_embeddings": [query_emb],
            "n_results": safe_n,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_params["where"] = where

        try:
            results = await asyncio.to_thread(
                lambda: self._collection.query(**query_params)
            )
        except Exception as exc:
            logger.error("search: ChromaDB query() failed: %s", exc, exc_info=True)
            return []

        formatted: List[Dict[str, Any]] = []
        if results and results.get("ids") and results["ids"][0]:
            for i, cid in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results.get("distances") else 0.0
                if distance >= similarity_threshold:
                    continue
                formatted.append({
                    "id": cid,
                    "chunk_text": results["documents"][0][i] if results.get("documents") else "",
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "distance": distance,
                })
                if len(formatted) >= n_results:
                    break

        latency = (time.time() - t0) * 1000
        logger.info(
            "search: returned %d results (threshold=%.2f, %.2f ms)",
            len(formatted), similarity_threshold, latency,
        )
        return formatted

    # -- Delete --

    async def delete_documents(self, ids: List[str]) -> bool:
        self._ensure_initialized()
        try:
            await asyncio.to_thread(lambda: self._collection.delete(ids=ids))
            logger.info("Deleted %d documents from ChromaDB", len(ids))
            return True
        except Exception as exc:
            logger.error("delete_documents failed: %s", exc)
            return False

    # -- List by source --

    async def get_documents_by_source(self, source: str) -> List[Dict[str, Any]]:
        self._ensure_initialized()
        try:
            results = await asyncio.to_thread(
                lambda: self._collection.get(
                    where={"source": source},
                    include=["metadatas"],
                )
            )
            return results.get("metadatas", []) or []
        except Exception as exc:
            logger.error("get_documents_by_source failed: %s", exc)
            return []

    # -- Stats --

    async def get_collection_stats(self) -> Dict[str, Any]:
        self._ensure_initialized()
        try:
            count = await asyncio.to_thread(self._collection.count)
            return {
                "collection_name": settings.CHROMADB_COLLECTION_NAME,
                "document_count": count,
                "embedding_dimension": self._ef.vector_dimension if self._ef else 0,
            }
        except Exception as exc:
            logger.error("get_collection_stats failed: %s", exc)
            return {
                "collection_name": settings.CHROMADB_COLLECTION_NAME,
                "document_count": 0,
                "error": str(exc),
            }