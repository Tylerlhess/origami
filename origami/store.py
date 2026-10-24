"""ContextStore — persistence layer for context segments."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .segment import Segment, SegmentStatus


class ContextStore:
    """Persistence layer for context segments and history."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize the store.
        
        Args:
            storage_path: Path to storage directory (None for in-memory only)
        """
        self.segments: Dict[str, Segment] = {}
        self.history: List[Dict[str, Any]] = []
        self.storage_path = Path(storage_path) if storage_path else None
        self._load()
    
    def _load(self):
        """Load segments from disk if storage_path is set."""
        if self.storage_path and self.storage_path.exists():
            try:
                with open(self.storage_path / "segments.json", "r") as f:
                    data = json.load(f)
                    for seg_dict in data.get("segments", []):
                        seg = Segment.from_dict(seg_dict)
                        self.segments[seg.id] = seg
            except Exception:
                pass
    
    def _save(self):
        """Save segments to disk if storage_path is set."""
        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            data = {
                "segments": [seg.to_dict() for seg in self.segments.values()],
                "history": self.history,
                "updated": time.time()
            }
            with open(self.storage_path / "segments.json", "w") as f:
                json.dump(data, f, indent=2)
    
    def add_segment(self, segment: Segment):
        """Add or update a segment."""
        self.segments[segment.id] = segment
        self._save()
    
    def remove_segment(self, seg_id: str):
        """Remove a segment."""
        if seg_id in self.segments:
            del self.segments[seg_id]
            self._save()
    
    def get_segment(self, seg_id: str) -> Optional[Segment]:
        """Get a single segment by ID."""
        return self.segments.get(seg_id)
    
    def get_all_segments(self) -> List[Segment]:
        """Get all segments."""
        return list(self.segments.values())
    
    def get_active_segments(self) -> List[Segment]:
        """Get all active segments (including contested, excluding false/banked/removed)."""
        return [s for s in self.segments.values() 
                if s.status in [SegmentStatus.ACTIVE, SegmentStatus.CONTESTED, SegmentStatus.ALTERED]]
    
    def log_action(self, action: str, details: Dict[str, Any]):
        """Log an action to history."""
        entry = {
            "timestamp": time.time(),
            "action": action,
            "details": details
        }
        self.history.append(entry)
        self._save()
    
    def clear(self):
        """Clear all segments."""
        self.segments.clear()
        self.history.clear()
        self._save()
    
    def get_context_set(self, max_tokens: int = 4096,
                        include_contested: bool = True,
                        include_false: bool = False) -> List[Segment]:
        """Get optimal context set based on priority scores.
        
        Args:
            max_tokens: Maximum tokens to include
            include_contested: Include segments marked as contested (disputed)
            include_false: Include segments marked as false (anti-facts)
            
        Returns:
            List of segments sorted by priority (highest first)
        """
        # Get all segments, filter by status
        all_segments = self.get_all_segments()
        eligible = []
        
        for seg in all_segments:
            if seg.status == SegmentStatus.FALSE and not include_false:
                continue  # Skip false segments by default
            if seg.status == SegmentStatus.CONTESTED and not include_contested:
                continue  # Skip contested segments if requested
            if seg.status == SegmentStatus.BANKED:
                continue  # Banked segments are separate
            if seg.status == SegmentStatus.REMOVED:
                continue  # Removed segments are gone
            eligible.append(seg)
        
        # Sort by priority (contested segments already have reduced priority)
        eligible.sort(key=lambda s: s.priority, reverse=True)
        
        selected = []
        token_count = 0
        
        for seg in eligible:
            seg_tokens = len(seg.content) // 4  # Rough token estimate
            if token_count + seg_tokens <= max_tokens:
                selected.append(seg)
                token_count += seg_tokens
            else:
                break
        
        return selected
