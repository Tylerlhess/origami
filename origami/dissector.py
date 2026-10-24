"""SegmentDissector — breaks raw conversation into segments."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .segment import Segment, SegmentType


class SegmentDissector:
    """Breaks raw conversation data into structured segments."""
    
    def dissect_messages(self, messages: List[Dict[str, Any]]) -> List[Segment]:
        """Dissect chat messages into segments.
        
        Args:
            messages: List of message dicts with role and content
            
        Returns:
            List of Segment objects
        """
        segments = []
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            msg_type = msg.get("type", "")
            
            if role == "user":
                seg_type = SegmentType.USER_MESSAGE
            elif role == "assistant":
                seg_type = SegmentType.ASSISTANT_MESSAGE
            elif role == "system":
                seg_type = SegmentType.SYSTEM
            elif msg_type == "tool_call":
                seg_type = SegmentType.TOOL_CALL
            elif msg_type == "tool_result":
                seg_type = SegmentType.TOOL_RESULT
            else:
                seg_type = SegmentType.USER_MESSAGE
            
            seg = Segment(
                type=seg_type,
                content=content,
                metadata=msg.get("metadata", {}),
                timestamp=time.time(),
            )
            seg.calculate_hash()
            segments.append(seg)
        
        return segments
    
    def dissect_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Segment]:
        """Dissect tool calls into segments.
        
        Args:
            tool_calls: List of tool call dicts
            
        Returns:
            List of Segment objects
        """
        segments = []
        
        for tc in tool_calls:
            name = tc.get("name", "")
            arguments = tc.get("arguments", {})
            result = tc.get("result", "")
            
            # Create tool call segment
            call_content = f"Tool: {name}\nArguments: {arguments}"
            call_seg = Segment(
                type=SegmentType.TOOL_CALL,
                content=call_content,
                metadata={"tool_name": name, "arguments": arguments},
            )
            call_seg.calculate_hash()
            segments.append(call_seg)
            
            # Create tool result segment if present
            if result:
                result_seg = Segment(
                    type=SegmentType.TOOL_RESULT,
                    content=result,
                    metadata={"tool_name": name},
                )
                result_seg.calculate_hash()
                segments.append(result_seg)
        
        return segments
    
    def dissect_system_notes(self, notes: List[str]) -> List[Segment]:
        """Dissect system notes into segments.
        
        Args:
            notes: List of system note strings
            
        Returns:
            List of Segment objects
        """
        segments = []
        for note in notes:
            seg = Segment(
                type=SegmentType.SYSTEM,
                content=note,
            )
            seg.calculate_hash()
            segments.append(seg)
        return segments
    
    def dissect_memory(self, memories: List[Dict[str, Any]]) -> List[Segment]:
        """Dissect memory entries into segments.
        
        Args:
            memories: List of memory dicts with key and value
            
        Returns:
            List of Segment objects
        """
        segments = []
        for mem in memories:
            key = mem.get("key", "")
            value = mem.get("value", "")
            seg = Segment(
                type=SegmentType.MEMORY,
                content=f"{key}: {value}",
                metadata={"memory_key": key},
            )
            seg.calculate_hash()
            segments.append(seg)
        return segments
    
    def find_duplicates(self, segments: List[Segment]) -> List[str]:
        """Find duplicate segments by content hash.
        
        Args:
            segments: List of segments to check
            
        Returns:
            List of segment IDs that are duplicates (keep first occurrence)
        """
        seen_hashes = set()
        duplicates = []
        
        for seg in segments:
            if seg.hash in seen_hashes:
                duplicates.append(seg.id)
            else:
                seen_hashes.add(seg.hash)
        
        return duplicates
