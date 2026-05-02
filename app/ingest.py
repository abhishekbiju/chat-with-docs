from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_pdf(path: Path) -> list[Document]:
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    for d in docs:
        d.metadata = dict(d.metadata)
        d.metadata["source"] = str(path)
        d.metadata.setdefault("kind", "pdf")
    return docs


def _load_text(path: Path) -> list[Document]:
    loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
    docs = loader.load()
    for d in docs:
        d.metadata = dict(d.metadata)
        d.metadata["source"] = str(path)
        d.metadata.setdefault("kind", path.suffix.lower().lstrip("."))
    return docs


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in {".txt", ".md", ".markdown"}:
        return _load_text(path)
    return []


def main() -> int:
    settings = get_settings()
    docs_dir = settings.documents_dir
    chroma_dir = settings.chroma_persist_dir

    if not docs_dir.is_dir():
        logger.error("Documents directory does not exist: %s", docs_dir)
        return 1

    logger.info("Scanning %s for PDF and text files…", docs_dir)
    all_docs: list[Document] = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        try:
            batch = load_file(path)
        except Exception:
            logger.exception("Failed to load %s — skipping", path)
            continue
        if batch:
            logger.info("Loaded %s (%d segment(s))", path.name, len(batch))
            all_docs.extend(batch)

    if not all_docs:
        logger.error("No supported documents found under %s", docs_dir)
        logger.error("Supported: .pdf, .txt, .md")
        return 1

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)
    logger.info("Split into %d chunk(s). Preparing embedding model…", len(chunks))

    try:
        embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    except Exception:
        logger.exception(
            "Could not load embedding model %r. Check network access to Hugging Face "
            "or set HF_HOME to a machine with the model cached.",
            settings.embedding_model,
        )
        return 1

    logger.info("Rebuilding Chroma index at %s …", chroma_dir)
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=str(chroma_dir),
    )
    logger.info("Ingestion complete. Vector store at %s", chroma_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
