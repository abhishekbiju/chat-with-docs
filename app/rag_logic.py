from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import Settings
from app.schemas import ChatTurn, QueryRequest, QueryResponse, SourceCitation, StreamDonePayload

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTIONS = """You are a careful assistant answering from the user's documents.
Rules:
- Base answers only on the provided CONTEXT. If context is insufficient, say you do not know or ask a clarifying question.
- Do not invent facts, citations, or page numbers not supported by CONTEXT.
- If the user refers to earlier messages, use CONVERSATION and CONTEXT together."""

USER_TEMPLATE = """CONTEXT (excerpts from the user's documents):
{context}

CONVERSATION (recent turns, may be empty):
{history_block}

QUESTION:
{question}
"""


def _excerpt(text: str, max_len: int = 400) -> str:
    t = text.strip().replace("\n", " ")
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def _page_from_metadata(meta: dict[str, Any]) -> int | None:
    for key in ("page", "page_label"):
        v = meta.get(key)
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)
    return None


def _source_file_from_metadata(meta: dict[str, Any]) -> str | None:
    src = meta.get("source")
    if not src:
        return None
    try:
        from pathlib import Path

        return Path(str(src)).name
    except Exception:
        return str(src)


def _format_history(history: list[ChatTurn]) -> str:
    if not history:
        return "(none)"
    lines: list[str] = []
    for turn in history[-12:]:
        role = turn.role.capitalize()
        lines.append(f"{role}: {turn.content.strip()}")
    return "\n".join(lines)


@dataclass
class RAGService:
    settings: Settings
    vectorstore: Chroma
    httpx_client: httpx.AsyncClient

    def index_size(self) -> int:
        """Number of vectors in Chroma, or -1 if unknown."""
        try:
            coll = self.vectorstore._collection  # noqa: SLF001
            return int(coll.count())
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Could not read Chroma collection size: %s", e)
            return -1

    def ensure_index_ready(self) -> None:
        n = self.index_size()
        if n == 0:
            raise ValueError(
                "The vector index is empty. Add PDFs or text files under documents/ "
                "and run: python -m app.ingest"
            )
        if n < 0:
            logger.debug("Collection size unknown; continuing.")

    def retrieve(
        self,
        query: str,
        k: int,
        use_mmr: bool,
        *,
        mmr_lambda_mult: float | None = None,
        mmr_fetch_k: int | None = None,
    ) -> list[tuple[Document, float | None]]:
        k = min(k, 20)
        if use_mmr:
            # MMR only reorders within the first `fetch_k` similarity hits. A hard cap that is
            # too small (previously 50) drops semantically relevant but lower-ranked chunks entirely.
            floor = max(self.settings.mmr_fetch_k, k * 4)
            if mmr_fetch_k is not None:
                floor = max(floor, mmr_fetch_k)
            fetch_cap = 200
            fetch_k = min(floor, fetch_cap)
            lam = (
                self.settings.mmr_lambda_mult
                if mmr_lambda_mult is None
                else mmr_lambda_mult
            )
            docs = self.vectorstore.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lam,
            )
            return [(d, None) for d in docs]

        pairs = self.vectorstore.similarity_search_with_score(query, k=k)
        return list(pairs)

    def build_sources(self, retrieved: list[tuple[Document, float | None]]) -> list[SourceCitation]:
        citations: list[SourceCitation] = []
        for doc, score in retrieved:
            meta = doc.metadata or {}
            citations.append(
                SourceCitation(
                    excerpt=_excerpt(doc.page_content),
                    source_file=_source_file_from_metadata(meta),
                    page=_page_from_metadata(meta),
                    relevance_score=float(score) if score is not None else None,
                )
            )
        return citations

    def _build_user_content(self, question: str, context_blocks: list[str], history: list[ChatTurn]) -> str:
        context = "\n---\n".join(context_blocks) if context_blocks else "(no matching passages found)"
        history_block = _format_history(history)
        return USER_TEMPLATE.format(
            context=context,
            history_block=history_block,
            question=question.strip(),
        )

    def _ollama_messages(self, user_content: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": user_content},
        ]

    async def generate_answer(
        self,
        question: str,
        context_blocks: list[str],
        history: list[ChatTurn],
    ) -> str:
        user_content = self._build_user_content(question, context_blocks, history)
        payload: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": self._ollama_messages(user_content),
            "stream": False,
        }
        try:
            response = await self.httpx_client.post(
                self.settings.ollama_chat_url,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            msg = data.get("message") or {}
            content = msg.get("content") or ""
            return str(content).strip() or "(empty model response)"
        except httpx.HTTPStatusError as e:
            logger.exception("Ollama HTTP error: %s", e)
            raise RuntimeError(
                f"Ollama returned HTTP {e.response.status_code}. "
                "Check that the model is pulled (e.g. `ollama pull mistral`) and OLLAMA_BASE_URL is correct."
            ) from e
        except httpx.RequestError as e:
            logger.exception("Ollama connection error: %s", e)
            raise RuntimeError(
                "Could not reach Ollama. Is it running? For Docker, set OLLAMA_BASE_URL "
                "(e.g. http://host.docker.internal:11434)."
            ) from e

    async def query(self, request: QueryRequest) -> QueryResponse:
        self.ensure_index_ready()
        retrieved = self.retrieve(
            request.query.strip(),
            request.k_retrieval,
            request.use_mmr,
            mmr_lambda_mult=request.mmr_lambda_mult,
            mmr_fetch_k=request.mmr_fetch_k,
        )
        context_blocks = [d.page_content for d, _ in retrieved]
        if not context_blocks:
            return QueryResponse(
                answer="I could not find relevant passages in your documents for that question. "
                "Try rephrasing or ingest more material.",
                sources=[],
            )
        answer = await self.generate_answer(request.query, context_blocks, request.history)
        sources = self.build_sources(retrieved)
        return QueryResponse(answer=answer, sources=sources)

    async def stream_query(self, request: QueryRequest) -> AsyncIterator[str]:
        """Server-Sent Events: `data: {"token": "..."}\\n\\n` then `event: done` with full payload."""
        self.ensure_index_ready()
        retrieved = self.retrieve(
            request.query.strip(),
            request.k_retrieval,
            request.use_mmr,
            mmr_lambda_mult=request.mmr_lambda_mult,
            mmr_fetch_k=request.mmr_fetch_k,
        )
        context_blocks = [d.page_content for d, _ in retrieved]
        sources = self.build_sources(retrieved)

        if not context_blocks:
            payload = StreamDonePayload(
                answer="I could not find relevant passages in your documents for that question.",
                sources=[],
            )
            yield f"event: done\ndata: {payload.model_dump_json()}\n\n"
            return

        user_content = self._build_user_content(request.query, context_blocks, request.history)
        stream_payload: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": self._ollama_messages(user_content),
            "stream": True,
        }
        parts: list[str] = []
        try:
            async with self.httpx_client.stream(
                "POST",
                self.settings.ollama_chat_url,
                json=stream_payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = (chunk.get("message") or {}).get("content") or ""
                    if token:
                        parts.append(token)
                        yield f"data: {json.dumps({'token': token})}\n\n"
                    if chunk.get("done"):
                        break
        except httpx.HTTPStatusError as e:
            err = json.dumps({"error": f"Ollama HTTP {e.response.status_code}"})
            yield f"event: error\ndata: {err}\n\n"
            return
        except httpx.RequestError as e:
            err = json.dumps({"error": f"Ollama unreachable: {e!s}"})
            yield f"event: error\ndata: {err}\n\n"
            return

        full = "".join(parts).strip()
        done = StreamDonePayload(answer=full or "(empty model response)", sources=sources)
        yield f"event: done\ndata: {done.model_dump_json()}\n\n"


def build_vectorstore(settings: Settings) -> Chroma:
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return Chroma(
        persist_directory=str(settings.chroma_persist_dir),
        embedding_function=embeddings,
    )


def build_http_client(settings: Settings) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(settings.httpx_timeout_seconds),
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
    )
