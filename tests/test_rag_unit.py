from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from app.config import Settings
from app.rag_logic import RAGService, _source_file_from_metadata, build_http_client


@pytest.mark.parametrize(
    "meta,expected",
    [
        ({}, None),
        ({"source": None}, None),
        ({"source": ""}, None),
        ({"other": 1}, None),
        ({"source": "/abs/path/to/file.txt"}, "file.txt"),
        ({"source": "weird"}, "weird"),
    ],
)
def test_source_file_from_metadata_messy_dicts(meta: dict, expected: str | None) -> None:
    assert _source_file_from_metadata(meta) == expected


@pytest.mark.asyncio
async def test_retrieve_mmr_reduces_duplicate_chunks_vs_similarity(tmp_path: Path) -> None:
    """MMR path should be able to return more diverse chunks than raw similarity (mocked)."""
    d = tmp_path / "docs"
    d.mkdir()
    settings = Settings(documents_dir=d, chroma_persist_dir=tmp_path / "c")
    dup = Document(page_content="identical boilerplate paragraph X", metadata={"source": "a.txt"})
    vs = MagicMock()
    vs.similarity_search_with_score.return_value = [(dup, 0.15)] * 6
    vs.max_marginal_relevance_search.return_value = [
        Document(page_content="identical boilerplate paragraph X", metadata={"source": "a.txt"}),
        Document(page_content="orthogonal topic about lunar regolith", metadata={"source": "b.txt"}),
        Document(page_content="third distinct angle on budgets", metadata={"source": "c.txt"}),
    ]

    async with build_http_client(settings) as client:
        rag = RAGService(settings=settings, vectorstore=vs, httpx_client=client)

        sim = rag.retrieve("query", k=5, use_mmr=False)
        mmr = rag.retrieve("query", k=5, use_mmr=True)

    vs.max_marginal_relevance_search.assert_called_once()
    mmr_kwargs = vs.max_marginal_relevance_search.call_args.kwargs
    assert mmr_kwargs["k"] == 5
    assert mmr_kwargs["fetch_k"] >= 5 * 4
    assert mmr_kwargs["lambda_mult"] == settings.mmr_lambda_mult

    assert len({doc.page_content for doc, _ in sim}) == 1
    assert len({doc.page_content for doc, _ in mmr}) == 3
