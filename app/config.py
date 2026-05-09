from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_chroma_dir() -> Path:
    return Path(__file__).resolve().parent / "chroma_db"


def _default_documents_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "documents"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434",
        description="Ollama server base URL (http or https, with host).",
    )
    ollama_chat_path: str = Field(default="/api/chat")
    llm_model: str = Field(default="mistral")
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    chroma_persist_dir: Path = Field(default_factory=_default_chroma_dir)
    documents_dir: Path = Field(default_factory=_default_documents_dir)
    httpx_timeout_seconds: float = Field(default=120.0, ge=5.0, le=600.0)
    mmr_fetch_k: int = Field(
        default=80,
        ge=8,
        le=200,
        description=(
            "Minimum similarity candidates to fetch before MMR reranking. "
            "MMR can only diversify within this pool—raise it when rare themes rank below the top few dozen hits."
        ),
    )
    mmr_lambda_mult: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description=(
            "MMR balance: higher → closer to pure similarity-to-query; lower → stronger diversity vs already-selected chunks. "
            "Default is below LangChain's 0.5 so thematic outliers (e.g. sustainability vs latency) can surface."
        ),
    )

    @property
    def ollama_chat_url(self) -> str:
        base = self.ollama_base_url.rstrip("/")
        path = self.ollama_chat_path if self.ollama_chat_path.startswith("/") else f"/{self.ollama_chat_path}"
        return f"{base}{path}"

    @field_validator("chroma_persist_dir", "documents_dir", mode="before")
    @classmethod
    def coerce_path(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()

    @field_validator("ollama_base_url")
    @classmethod
    def ollama_base_url_must_be_http(cls, v: str) -> str:
        raw = (v or "").strip()
        parsed = urlparse(raw)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(
                "ollama_base_url must be an http or https URL with a host "
                "(e.g. http://127.0.0.1:11434)"
            )
        return raw

    @field_validator("documents_dir", mode="after")
    @classmethod
    def documents_dir_must_exist(cls, v: Path) -> Path:
        if not v.is_dir():
            raise ValueError(f"documents_dir must be an existing directory: {v}")
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
