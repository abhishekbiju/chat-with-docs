from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_settings_rejects_malformed_ollama_base_url(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    for bad in ("not-a-url", "ftp://127.0.0.1:11434", "http://", "", "://oops"):
        with pytest.raises(ValidationError) as exc:
            Settings(ollama_base_url=bad, documents_dir=docs)
        errs = exc.value.errors()
        assert any(e.get("loc") == ("ollama_base_url",) for e in errs)


def test_settings_rejects_missing_documents_dir(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    assert not missing.exists()
    with pytest.raises(ValidationError) as exc:
        Settings(documents_dir=missing)
    errs = exc.value.errors()
    assert any("documents_dir" in str(e.get("loc", ())) for e in errs)


def test_settings_accepts_valid_http_urls(tmp_path: Path) -> None:
    d = tmp_path / "docs"
    d.mkdir()
    s = Settings(ollama_base_url="http://127.0.0.1:11434", documents_dir=d)
    assert s.ollama_chat_url.startswith("http://127.0.0.1:11434")

    s2 = Settings(ollama_base_url="https://ollama.example.com", documents_dir=d)
    assert "https://ollama.example.com" in s2.ollama_chat_url
