from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.config import Settings
from app.rag_logic import build_vectorstore


def _retrieved_chunk_ids(vectorstore, question: str, k: int = 3) -> list[str]:
    pairs = vectorstore.similarity_search_with_score(question, k=k)
    out: list[str] = []
    for doc, _score in pairs:
        if doc.id:
            out.append(str(doc.id))
    return out


@pytest.fixture(scope="module")
def rag_eval_vectorstore():
    base = Path(tempfile.mkdtemp(prefix="rag_eval_"))
    docs_dir = base / "documents"
    chroma_dir = base / "chroma"
    docs_dir.mkdir(parents=True)
    chroma_dir.mkdir(parents=True)

    settings = Settings(documents_dir=docs_dir, chroma_persist_dir=chroma_dir)

    cases_path = Path(__file__).resolve().parent / "fixtures" / "rag_eval_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))

    contents = [
        "CONFIDENTIAL AURORA: The Aurora plumbus codeword phrase is plumbus-seven-aurora-nine.",
        "TETHYS STATION: The Tethys station override PIN digits are 8801 for emergency airlock.",
        "INVENTORY 7B: The banana inventory count for warehouse 7B is exactly 1442 crates on Jan 3.",
        "LUNAR MEMO: Dr. Chen signed the memo about lunar dust protocols on page 2 of LDM-14.",
        "REFUNDS: The refund policy window for titanium widgets is 30 days with receipt and RMA.",
    ]
    ids = ["eval_aurora", "eval_tethys", "eval_banana", "eval_lunar", "eval_refund"]
    assert len(contents) == len(ids) == len(cases)

    documents = [
        Document(page_content=text, metadata={"source": f"{id_}.txt"}, id=id_)
        for text, id_ in zip(contents, ids, strict=True)
    ]

    vs = build_vectorstore(settings)
    vs.add_documents(documents, ids=ids)
    try:
        yield vs, cases
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_rag_eval_ground_truth_chunk_ids(rag_eval_vectorstore) -> None:
    vectorstore, cases = rag_eval_vectorstore
    for row in cases:
        q = row["question"]
        expected: list[str] = row["expected_chunk_ids"]
        retrieved = _retrieved_chunk_ids(vectorstore, q, k=5)
        for chunk_id in expected:
            assert chunk_id in retrieved, (
                f"Expected chunk {chunk_id!r} in retrieval for {q!r}, got {retrieved!r}"
            )
