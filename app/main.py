from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.rag_logic import RAGService, build_http_client, build_vectorstore
from app.schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    vectorstore = build_vectorstore(settings)
    async with build_http_client(settings) as client:
        app.state.rag = RAGService(settings=settings, vectorstore=vectorstore, httpx_client=client)
        logger.info("RAG service ready (Chroma at %s)", settings.chroma_persist_dir)
        yield


app = FastAPI(
    title="Chat with Docs",
    description="RAG API over your PDFs and text files, backed by Ollama and Chroma.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready(request: Request) -> dict[str, str | int | bool]:
    """Checks vector index and Ollama reachability."""
    settings = get_settings()
    rag: RAGService = request.app.state.rag
    index_count = rag.index_size()
    ollama_ok = False
    tags_url = settings.ollama_base_url.rstrip("/") + "/api/tags"
    try:
        async with httpx.AsyncClient(timeout=5.0) as probe:
            r = await probe.get(tags_url)
            ollama_ok = r.is_success
    except httpx.RequestError:
        ollama_ok = False

    return {
        "index_documents": index_count,
        "ollama_reachable": ollama_ok,
        "ready": ollama_ok and index_count > 0,
    }


@app.get("/api/info")
async def api_info(request: Request) -> dict[str, str | int]:
    settings = get_settings()
    rag: RAGService = request.app.state.rag
    return {
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "ollama_base_url": settings.ollama_base_url,
        "indexed_chunks": rag.index_size(),
    }


@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest, req: Request) -> QueryResponse:
    rag: RAGService = req.app.state.rag
    try:
        return await rag.query(request)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


@app.post("/chat/stream")
async def chat_stream(request: QueryRequest, req: Request) -> StreamingResponse:
    rag: RAGService = req.app.state.rag

    async def event_generator():
        try:
            async for chunk in rag.stream_query(request):
                yield chunk
        except ValueError as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/")
async def serve_ui():
    index = STATIC_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {"message": "UI not installed. Open /docs for the API.", "docs": "/docs"}

