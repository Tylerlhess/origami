"""Origami context engine — Hermes plugin entry point."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine, EngineConfig

from .config import OrigamiConfig
from .manager import ContextManager

logger = logging.getLogger(__name__)

class OrigamiContextEngine(ContextEngine):
    """Context distillation engine that dissects, grades, and banks segments."""

    def __init__(self, config: EngineConfig):
        self._config = config
        self._origami_config = self._build_origami_config()
        self._manager: Optional[ContextManager] = None
        self._bank: Dict[str, Any] = {}
        self._state = ""
        self._initialized = False

    def _build_origami_config(self) -> OrigamiConfig:
        """Build OrigamiConfig from EngineConfig and environment."""
        import agent.config as agent_config
        return OrigamiConfig(
            systemone_base_url=os.getenv("ORIGAMI_SYSTEMONE_URL", "http://127.0.0.1:8005"),
            systemone_api_key=os.getenv("ORIGAMI_SYSTEMONE_API", "token-abc123"),
            chat_base_url=os.getenv("ORIGAMI_CHAT_URL", "http://127.0.0.1:8005"),
            chat_api_key=os.getenv("ORIGAMI_CHAT_API", "token-abc123"),
            model=self._config.model,
            storage_path=os.getenv("ORIGAMI_STORAGE", None),
        )

    def _get_manager(self) -> ContextManager:
        if self._manager is None:
            try:
                self._manager = ContextManager(config=self._origami_config)
                self._initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize origami engine: {e}")
                self._manager = None
        return self._manager

    def compress(self, messages: List[Dict], system_prompt: str,
                 *, reserve_tokens: Optional[int] = None,
                 focus_topics: Optional[List[str]] = None,
                 user_instruction: Optional[str] = None,
                 api_call: Any = None,
                 session_id: Optional[str] = None,
                 platform: Optional[str] = None,
                **_kwargs: Any) -> str:
        """Compress messages via origami distillation."""
        start = time.time()
        manager = self._get_manager()

        if not manager:
            return self._fallback_compress(messages)

        try:
            # Extract state from messages
            state = self._extract_state(messages, user_instruction)

            # Add segments to origami
            origami_messages = self._convert_messages(messages)
            manager.add(
                messages=origami_messages,
                system_notes=[system_prompt] if system_prompt else None,
                state=state
            )

            # Distill and get context
            max_tokens = reserve_tokens or 4096
            result = manager.distill(max_tokens=max_tokens)

            # Get the distilled context
            segments = manager.get_context(max_tokens=max_tokens)
            compressed = self._segments_to_text(segments)

            elapsed = time.time() - start
            logger.info(
                f"Origami compressed {len(messages)} messages -> "
                f"{len(segments)} segments in {elapsed:.2f}s"
            )
            return compressed

        except Exception as e:
            logger.error(f"Origami compression failed, falling back: {e}")
            return self._fallback_compress(messages)

    def _extract_state(self, messages: List[Dict], instruction: Optional[str]) -> str:
        """Extract current state from messages and instruction."""
        parts = []
        if instruction:
            parts.append(instruction)
        # Use last few messages for context
        for msg in messages[-5:]:
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                parts.append(content[:200])
        return " ".join(parts)

    def _convert_messages(self, messages: List[Dict]) -> List[Dict]:
        """Convert messages to origami format."""
        return messages

    def _segments_to_text(self, segments: List[Any]) -> str:
        """Convert origami segments to text."""
        parts = []
        for seg in segments:
            content = seg.content if hasattr(seg, "content") else str(seg)
            parts.append(content)
        return "\n\n".join(parts)

    def _fallback_compress(self, messages: List[Dict]) -> str:
        """Fallback compression: simple truncation."""
        parts = []
        for msg in messages[-20:]:
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content[:500])
        return "\n\n".join(parts)

# Module-level registration function expected by Hermes plugin loader
_origami_engine: Optional[OrigamiContextEngine] = None

def register_context_engine() -> OrigamiContextEngine:
    """Register the origami context engine.
    
    Called by the Hermes plugin loader. Returns the engine instance.
    """
    global _origami_engine
    if _origami_engine is None:
        _origami_engine = OrigamiContextEngine
    return _origami_engine
