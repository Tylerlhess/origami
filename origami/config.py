"""OrigamiConfig — Configuration for origami context distillation service."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OrigamiConfig:
    """Configuration for the origami context distiller.
    
    Supports separate endpoints for SystemOne (decision) calls vs 
    chat/completion (LLM) calls.
    
    Example:
        config = OrigamiConfig(
            systemone_base_url="http://127.0.0.1:8005",
            systemone_endpoint="/v1/systemone",
            systemone_api_key="token-abc123",
            chat_base_url="http://127.0.0.1:8080",
            chat_endpoint="/v1/chat/completions",
            chat_api_key="sk-...",
            model="Qwopus3.8-27B-Flash-Q4_K_M"
        )
    """

    # SystemOne (decision/scoring) endpoint configuration
    systemone_base_url: str = field(default="http://127.0.0.1:8005")
    systemone_endpoint: str = field(default="/v1/systemone")
    systemone_api_key: str = field(default="token-abc123")
    systemone_timeout: float = field(default=30.0)

    # Chat/Completion (LLM) endpoint configuration
    chat_base_url: str = field(default="http://127.0.0.1:8005")
    chat_endpoint: str = field(default="/v1/chat/completions")
    chat_api_key: str = field(default="token-abc123")

    # Model configuration
    model: str = field(default="Qwopus3.8-27B-Flash-Q4_K_M")

    # Storage
    storage_path: Optional[str] = field(default=None)

    # Scoring thresholds
    truthfulness_threshold: float = field(default=0.7)
    contested_threshold: float = field(default=0.3)
    false_threshold: float = field(default=0.1)
    recall_relevance_threshold: float = field(default=0.3)
    bank_irrelevance_threshold: float = field(default=0.2)

    def get_systemone_url(self) -> str:
        """Get full SystemOne API URL."""
        return f"{self.systemone_base_url.rstrip('/')}{self.systemone_endpoint}"

    def get_chat_url(self) -> str:
        """Get full chat/completions URL."""
        return f"{self.chat_base_url.rstrip('/')}{self.chat_endpoint}"

    @classmethod
    def from_env(cls) -> "OrigamiConfig":
        """Create config from environment variables.
        
        Environment variables:
            ORIGAMI_SYSTEMONE_URL — SystemOne base URL
            ORIGAMI_SYSTEMONE_API — SystemOne API key
            ORIGAMI_CHAT_URL — Chat/completions base URL
            ORIGAMI_CHAT_API — Chat/completions API key
            ORIGAMI_MODEL — Model name
            ORIGAMI_STORAGE — Storage path
        """
        return cls(
            systemone_base_url=os.getenv("ORIGAMI_SYSTEMONE_URL", "http://127.0.0.1:8005"),
            systemone_api_key=os.getenv("ORIGAMI_SYSTEMONE_API", "token-abc123"),
            chat_base_url=os.getenv("ORIGAMI_CHAT_URL", "http://127.0.0.1:8005"),
            chat_api_key=os.getenv("ORIGAMI_CHAT_API", "token-abc123"),
            model=os.getenv("ORIGAMI_MODEL", "Qwopus3.8-27B-Flash-Q4_K_M"),
            storage_path=os.getenv("ORIGAMI_STORAGE", None),
        )
