"""Runtime settings, read from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()  # reads a local .env (never committed); real environment variables take precedence


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    # "heuristic" = rule-based client, no LLM needed (tests, CI, offline).
    # "ollama" | "openai" | "anthropic" = real LLM drives the agents.
    llm_provider: str = field(default_factory=lambda: _env("DATASCOUT_LLM_PROVIDER", "heuristic"))
    llm_model: str = field(default_factory=lambda: _env("DATASCOUT_LLM_MODEL", "qwen2.5:7b"))
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    # any OpenAI-compatible endpoint (Groq, OpenRouter, vLLM, LM Studio...); key comes from OPENAI_API_KEY
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL", ""))
    llm_timeout_s: float = field(default_factory=lambda: float(_env("DATASCOUT_LLM_TIMEOUT_S", "120")))
    llm_max_retries: int = field(default_factory=lambda: int(_env("DATASCOUT_LLM_MAX_RETRIES", "3")))

    columns_per_batch: int = 20          # profiles per LLM call (context management)
    max_prep_retries: int = 2            # validator -> preparer retry budget
    max_upload_mb: int = field(default_factory=lambda: int(_env("DATASCOUT_MAX_UPLOAD_MB", "25")))
    data_dir: str = field(default_factory=lambda: _env("DATASCOUT_DATA_DIR", "./data"))

    @property
    def langfuse_enabled(self) -> bool:
        return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def get_settings() -> Settings:
    return Settings()
