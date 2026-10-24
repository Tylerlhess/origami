"""SystemOneGrader — uses SystemOne API to score segments."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional, Tuple

import requests

from .config import OrigamiConfig

logger = logging.getLogger(__name__)


class SystemOneGrader:
    """Grades segments using SystemOne API for truthfulness and usefulness.
    
    Supports separate SystemOne (decision) endpoints from chat/completion endpoints
    via OrigamiConfig.
    """
    
    def __init__(self, 
                 base_url: str = "http://127.0.0.1:8005",
                 api_key: str = "token-abc123",
                 timeout: float = 30.0,
                 model: str = "Qwopus3.8-27B-Flash-Q4_K_M",
                 config: Optional[OrigamiConfig] = None):
        """Initialize the grader.
        
        Args:
            base_url: SystemOne base URL
            api_key: API key for authentication
            timeout: Request timeout in seconds
            model: Model name for SystemOne
            config: Optional OrigamiConfig for full configuration (overrides individual args)
        """
        if config:
            self.base_url = config.systemone_base_url.rstrip("/")
            self.systemone_endpoint = config.systemone_endpoint
            self.api_key = config.systemone_api_key
            self.timeout = config.systemone_timeout
            self.model = config.model
        else:
            self.base_url = base_url.rstrip("/")
            self.systemone_endpoint = "/v1/systemone"
            self.api_key = api_key
            self.timeout = timeout
            self.model = model
        
        self.available = True
        self._last_error = None
    
    def _systemone_url(self) -> str:
        """Get the full SystemOne API URL."""
        return f"{self.base_url}{self.systemone_endpoint}"
    
    def _systemone_request(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make a request to SystemOne API.
        
        Args:
            payload: Request payload
            
        Returns:
            Response JSON or None if failed
        """
        if not self.available:
            return None
        
        url = self._systemone_url()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Add model to payload (required by SystemOne)
        payload = dict(payload)
        payload["model"] = self.model
        
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"SystemOne request failed: {e}")
            self._last_error = str(e)
            self.available = False
            return None
    
    def grade_truthfulness(self, state: str, content: str) -> float:
        """Grade truthfulness of content.
        
        Args:
            state: Context/state for the decision
            content: Content to grade
            
        Returns:
            Truthfulness score 0.0-1.0 (0.5 if unavailable)
        """
        payload = {
            "state": state,
            "questions": {
                "truthfulness": {
                    "type": "noul",
                    "instructions": (
                        "Is this statement truthful and accurate? "
                        "Consider factual correctness, logical consistency, "
                        "and absence of hallucinations. "
                        "Return a score between 0 (completely false) and 1 (completely true)."
                    )
                }
            }
        }
        
        result = self._systemone_request(payload)
        if result:
            answers = result.get("answers", {})
            truth = answers.get("truthfulness", {})
            noul = truth.get("noul", 0.5)
            return max(0.0, min(1.0, noul))
        
        # Fallback heuristic: content length and format
        return self._fallback_truthfulness(content)
    
    def grade_usefulness(self, state: str, content: str, context: str = "") -> float:
        """Grade usefulness of content for current context.
        
        Args:
            state: Current state/context
            content: Content to grade
            context: Additional context for relevance
            
        Returns:
            Usefulness score 0.0-1.0 (0.5 if unavailable)
        """
        instructions = (
            "How useful is this information for answering the current query? "
            "Consider relevance, specificity, and actionability. "
            "Return a score between 0 (not useful) and 1 (extremely useful)."
        )
        
        if context:
            instructions += f" Additional context: {context[:200]}"
        
        payload = {
            "state": state,
            "questions": {
                "usefulness": {
                    "type": "noul",
                    "instructions": instructions
                }
            }
        }
        
        result = self._systemone_request(payload)
        if result:
            answers = result.get("answers", {})
            useful = answers.get("usefulness", {})
            noul = useful.get("noul", 0.5)
            return max(0.0, min(1.0, noul))
        
        # Fallback heuristic: keyword overlap with state
        return self._fallback_usefulness(content, state)
    
    def classify_relevance(self, state: str, content: str) -> Tuple[str, Dict[str, float]]:
        """Classify relevance category and return probabilities.
        
        Args:
            state: Current state
            content: Content to classify
            
        Returns:
            Tuple of (category, probabilities_dict)
        """
        payload = {
            "state": state,
            "questions": {
                "relevance": {
                    "type": "choice",
                    "instructions": "Classify the relevance of this content",
                    "criteria": {
                        "A": "Critical - essential for answering the query",
                        "B": "Important - significantly helps answer the query",
                        "C": "Moderate - somewhat relevant",
                        "D": "Low - peripheral or background info",
                        "E": "Irrelevant - not related to the query"
                    }
                }
            }
        }
        
        result = self._systemone_request(payload)
        if result:
            answers = result.get("answers", {})
            rel = answers.get("relevance", {})
            choice = rel.get("choice", "C")
            probs = rel.get("probabilities", {})
            return choice, probs
        
        # Fallback: return neutral
        return "C", {"A": 0.0, "B": 0.0, "C": 0.6, "D": 0.3, "E": 0.1}
    
    def _fallback_truthfulness(self, content: str) -> float:
        """Fallback heuristic for truthfulness when SystemOne unavailable."""
        # Simple heuristic: longer content is more likely to be detailed/true
        length_score = min(1.0, len(content) / 500.0)
        
        # Penalize very short content
        if len(content) < 20:
            return 0.3
        
        return 0.5 + length_score * 0.3
    
    def _fallback_usefulness(self, content: str, state: str) -> float:
        """Fallback heuristic for usefulness when SystemOne unavailable."""
        # Simple keyword overlap
        content_words = set(content.lower().split())
        state_words = set(state.lower().split())
        overlap = len(content_words & state_words)
        max_words = max(len(content_words), len(state_words), 1)
        overlap_score = overlap / max_words
        
        return 0.3 + overlap_score * 0.5
