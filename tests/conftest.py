from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from app.config import Settings, get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(scope="module")
def integration_dirs() -> Path:
    base = Path(tempfile.mkdtemp(prefix="chat_with_docs_e2e_"))
    docs = base / "documents"
    chroma = base / "chroma"
    docs.mkdir(parents=True)
    chroma.mkdir(parents=True)
    (docs / "stream_seed.txt").write_text(
        "STREAM_TEST_CONTEXT The recovery code for vault nine is sapphire-moon-441.\n",
        encoding="utf-8",
    )
    return base


@pytest.fixture(scope="module")
def integration_client(integration_dirs: Path):
    preserved_doc = os.environ.get("DOCUMENTS_DIR")
    preserved_chroma = os.environ.get("CHROMA_PERSIST_DIR")
    os.environ["DOCUMENTS_DIR"] = str(integration_dirs / "documents")
    os.environ["CHROMA_PERSIST_DIR"] = str(integration_dirs / "chroma")
    get_settings.cache_clear()

    from langchain_core.documents import Document

    from app.main import app
    from app.rag_logic import build_vectorstore

    settings = Settings()
    vs = build_vectorstore(settings)
    vs.add_texts(
        ["STREAM_TEST_CONTEXT The recovery code for vault nine is sapphire-moon-441."],
        metadatas=[{"source": str(integration_dirs / "documents" / "stream_seed.txt")}],
        ids=["chunk_stream_seed"],
    )
    del vs

    from fastapi.testclient import TestClient

    try:
        with TestClient(app) as client:
            yield client
    finally:
        if preserved_doc is None:
            os.environ.pop("DOCUMENTS_DIR", None)
        else:
            os.environ["DOCUMENTS_DIR"] = preserved_doc
        if preserved_chroma is None:
            os.environ.pop("CHROMA_PERSIST_DIR", None)
        else:
            os.environ["CHROMA_PERSIST_DIR"] = preserved_chroma
        get_settings.cache_clear()
        shutil.rmtree(integration_dirs, ignore_errors=True)
