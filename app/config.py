from functools import lru_cache
from pathlib import Path

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
        description="Ollama server base URL (no trailing slash).",
    )
    ollama_chat_path: str = Field(default="/api/chat")
    llm_model: str = Field(default="mistral")
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    chroma_persist_dir: Path = Field(default_factory=_default_chroma_dir)
    documents_dir: Path = Field(default_factory=_default_documents_dir)
    httpx_timeout_seconds: float = Field(default=120.0, ge=5.0, le=600.0)
    mmr_fetch_k: int = Field(default=20, ge=4, le=100)

    @property
    def ollama_chat_url(self) -> str:
        base = self.ollama_base_url.rstrip("/")
        path = self.ollama_chat_path if self.ollama_chat_path.startswith("/") else f"/{self.ollama_chat_path}"
        return f"{base}{path}"

    @field_validator("chroma_persist_dir", "documents_dir", mode="before")
    @classmethod
    def coerce_path(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
