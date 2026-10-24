"""Segment dataclass — represents one chunk of context."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class SegmentType(Enum):
    """Types of context segments."""
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SYSTEM = "system"
    MEMORY = "memory"
    INSTRUCTION = "instruction"


class SegmentStatus(Enum):
    """Status of a segment in the context."""
    ACTIVE = "active"
    ALTERED = "altered"
    BANKED = "banked"       # Parked outside context, recalled when relevant
    CONTESTED = "contested" # Marked as disputed/unverified (gray zone)
    FALSE = "false"         # Confirmed wrong (anti-fact), excluded from context
    REMOVED = "removed"     # Deleted entirely


@dataclass
class Segment:
    """A single chunk of context data.
    
    Attributes:
        id: Unique identifier for the segment
        type: Type of content (message, tool_call, etc.)
        content: The actual text/data
        timestamp: Unix timestamp when created
        truthfulness_score: 0.0-1.0 (graded by SystemOne)
        usefulness_score: 0.0-1.0 (graded by SystemOne)
        priority: Calculated score for ordering (weighted combination)
        status: Current status (active, altered, removed)
        metadata: Additional data (tool name, function args, etc.)
        hash: Content hash for duplicate detection
    """
    id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])
    type: SegmentType = SegmentType.USER_MESSAGE
    content: str = ""
    timestamp: float = field(default_factory=time.time)
    truthfulness_score: float = 0.0
    usefulness_score: float = 0.0
    priority: float = 0.0
    status: SegmentStatus = SegmentStatus.ACTIVE
    metadata: dict = field(default_factory=dict)
    hash: str = field(default="", compare=False)
    
    def calculate_hash(self) -> str:
        """Calculate content hash for duplicate detection."""
        self.hash = hashlib.md5(self.content.encode()).hexdigest()
        return self.hash
    
    def calculate_priority(self, weight_truthfulness: float = 0.3, 
                          weight_usefulness: float = 0.5,
                          weight_recency: float = 0.2) -> float:
        """Calculate priority score based on scores and age.
        
        Accounts for contested/false status — disputed segments get a
        reduced priority multiplier so they don't rank high.
        
        Args:
            weight_truthfulness: Weight for truthfulness (0-1)
            weight_usefulness: Weight for usefulness (0-1)
            weight_recency: Weight for recency (0-1)
            
        Returns:
            Priority score (0-1)
        """
        # Calculate recency score (newer = higher)
        age_hours = (time.time() - self.timestamp) / 3600
        recency_score = max(0.0, 1.0 - age_hours / 24)  # Half-life of 24 hours
        
        # Weighted combination
        base_priority = (
            self.truthfulness_score * weight_truthfulness +
            self.usefulness_score * weight_usefulness +
            recency_score * weight_recency
        )
        
        # Apply status multiplier
        status_multiplier = 1.0
        if self.status == SegmentStatus.CONTESTED:
            status_multiplier = 0.5  # Disputed — reduce priority by 50%
        elif self.status == SegmentStatus.FALSE:
            status_multiplier = 0.0  # Confirmed wrong — no priority
        
        self.priority = base_priority * status_multiplier
        return self.priority
    
    def mark_contested(self, reason: str, confidence: float = 0.5):
        """Mark this segment as contested (disputed/unverified).
        
        Args:
            reason: Why it's contested
            confidence: How confident we are it's wrong (0-1)
                       0.0 = definitely wrong, 1.0 = definitely right
        """
        self.status = SegmentStatus.CONTESTED
        self.metadata["contested_reason"] = reason
        self.metadata["contested_confidence"] = confidence
        self.metadata["contested_at"] = time.time()
        self.calculate_priority()
    
    def mark_false(self, reason: str, corrected_content: Optional[str] = None):
        """Mark this segment as confirmed false (anti-fact).
        
        Args:
            reason: Why it's false
            corrected_content: Optional corrected version of the content
        """
        self.status = SegmentStatus.FALSE
        self.metadata["false_reason"] = reason
        if corrected_content:
            self.metadata["corrected_content"] = corrected_content
        self.metadata["marked_false_at"] = time.time()
        self.calculate_priority()
    
    def mark_verified(self, verified_by: str = "systemone"):
        """Mark this segment as verified correct.
        
        Args:
            verified_by: Who verified it
        """
        self.status = SegmentStatus.ACTIVE
        self.metadata["verified_at"] = time.time()
        self.metadata["verified_by"] = verified_by
        self.calculate_priority()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "truthfulness_score": self.truthfulness_score,
            "usefulness_score": self.usefulness_score,
            "priority": self.priority,
            "status": self.status.value,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Segment":
        """Create Segment from dictionary."""
        seg = cls(
            id=data.get("id", ""),
            type=SegmentType(data.get("type", "user_message")),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", time.time()),
            truthfulness_score=data.get("truthfulness_score", 0.0),
            usefulness_score=data.get("usefulness_score", 0.0),
            priority=data.get("priority", 0.0),
            status=SegmentStatus(data.get("status", "active")),
            metadata=data.get("metadata", {}),
        )
        seg.calculate_hash()
        return seg
