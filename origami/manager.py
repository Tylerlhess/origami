"""ContextManager — main service that orchestrates the context distillation flow."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .config import OrigamiConfig
from .dissector import SegmentDissector
from .grader import SystemOneGrader
from .segment import Segment, SegmentStatus, SegmentType
from .store import ContextStore

logger = logging.getLogger(__name__)


class ContextManager:
    """Main service for context distillation and management."""
    
    def __init__(self,
                 config: Optional[OrigamiConfig] = None,
                 systemone_url: str = "http://127.0.0.1:8005",
                 storage_path: Optional[str] = None,
                 model: str = "Qwopus3.8-27B-Flash-Q4_K_M"):
        """Initialize the context manager.
        
        Args:
            config: Optional OrigamiConfig for full configuration
                   (overrides individual args if provided)
            systemone_url: SystemOne API URL (used if config not provided)
            storage_path: Path for persistence (None for in-memory)
            model: SystemOne model name (used if config not provided)
        """
        if config:
            self.config = config
            self.dissector = SegmentDissector()
            self.grader = SystemOneGrader(config=config)
            self.store = ContextStore(storage_path=config.storage_path)
        else:
            self.config = OrigamiConfig(
                systemone_base_url=systemone_url,
                model=model,
                storage_path=storage_path
            )
            self.dissector = SegmentDissector()
            self.grader = SystemOneGrader(base_url=systemone_url, model=model)
            self.store = ContextStore(storage_path=storage_path)
        self.current_state = ""
        self.bank: Dict[str, Segment] = {}  # Parked segments outside conversation
    
    def add(self,
            messages: Optional[List[Dict[str, Any]]] = None,
            tool_calls: Optional[List[Dict[str, Any]]] = None,
            system_notes: Optional[List[str]] = None,
            memories: Optional[List[Dict[str, Any]]] = None,
            state: Optional[str] = None):
        """Add conversation data to the context.
        
        Args:
            messages: Chat messages
            tool_calls: Tool call results
            system_notes: System notes/instructions
            memories: Memory entries
            state: Current state/context for grading
        """
        if state:
            self.current_state = state
        
        segments = []
        
        if messages:
            segments.extend(self.dissector.dissect_messages(messages))
        
        if tool_calls:
            segments.extend(self.dissector.dissect_tool_calls(tool_calls))
        
        if system_notes:
            segments.extend(self.dissector.dissect_system_notes(system_notes))
        
        if memories:
            segments.extend(self.dissector.dissect_memory(memories))
        
        # Remove duplicates
        duplicate_ids = self.dissector.find_duplicates(segments)
        segments = [s for s in segments if s.id not in duplicate_ids]
        
        # Add to store
        for seg in segments:
            self.store.add_segment(seg)
        
        logger.info(f"Added {len(segments)} segments to context")
    
    def add_instruction(self, content: str, task_id: str,
                        tags: Optional[List[str]] = None):
        """Add a task-specific instruction that is kept banked until the task is active.
        
        This allows instructions to be parked outside the conversation and
        recalled when the matching task becomes relevant.
        
        Args:
            content: Instruction text
            task_id: Task identifier for recall
            tags: Optional tags for additional filtering
        """
        from .segment import Segment, SegmentType
        
        seg = Segment(
            type=SegmentType.INSTRUCTION,
            content=content,
            metadata={
                "task_id": task_id,
                "tags": tags or [],
                "task_description": content[:100]
            }
        )
        
        # Start in banked state — parked until task is active
        seg.status = SegmentStatus.BANKED
        self.bank[seg.id] = seg
        
        logger.info(f"Banked instruction for task {task_id}")
    
    def distill(self, max_tokens: int = 4096) -> Dict[str, Any]:
        """Grade, order, and compact segments.
        
        Segments that don't fit the token budget are BANKED (parked outside
        the conversation) rather than deleted. They can be recalled later via
        recall_bank() when the state changes.
        
        Args:
            max_tokens: Maximum tokens to retain
            
        Returns:
            Distillation report with scores and banked segments
        """
        active_segments = self.store.get_active_segments()
        
        # Grade each segment
        graded = []
        for seg in active_segments:
            # Grade truthfulness
            truthfulness = self.grader.grade_truthfulness(self.current_state, seg.content)
            seg.truthfulness_score = truthfulness
            
            # Grade usefulness
            usefulness = self.grader.grade_usefulness(self.current_state, seg.content)
            seg.usefulness_score = usefulness
            
            # Calculate priority
            seg.calculate_priority()
            
            graded.append(seg)
        
        # Sort by priority
        graded.sort(key=lambda s: s.priority, reverse=True)
        
        # Determine which segments to keep based on token budget
        kept = []
        banked = []
        token_count = 0
        
        for seg in graded:
            seg_tokens = len(seg.content) // 4
            if token_count + seg_tokens <= max_tokens:
                kept.append(seg)
                token_count += seg_tokens
            else:
                seg.status = SegmentStatus.BANKED
                seg.metadata["banked_at"] = time.time()
                seg.metadata["banked_reason"] = "token_budget"
                self.bank[seg.id] = seg
                banked.append(seg)
                self.store.add_segment(seg)  # Update status
        
        # Update kept segments
        for seg in kept:
            self.store.add_segment(seg)
        
        # Log action
        self.store.log_action("distill", {
            "kept": len(kept),
            "banked": len(banked),
            "tokens": token_count
        })
        
        return {
            "kept": [s.id for s in kept],
            "banked": [s.id for s in banked],
            "token_count": token_count,
            "scores": {s.id: {"truthfulness": s.truthfulness_score, 
                             "usefulness": s.usefulness_score,
                             "priority": s.priority} 
                       for s in kept}
        }
    
    def get_context(self, max_tokens: int = 4096) -> List[Segment]:
        """Get the optimized context set for the next turn.
        
        Args:
            max_tokens: Maximum tokens to include
            
        Returns:
            List of segments sorted by priority
        """
        return self.store.get_context_set(max_tokens)
    
    def recall_bank(self, state: Optional[str] = None,
                     min_relevance: Optional[float] = None) -> List[Segment]:
        """Recall banked segments that are relevant to the current state.
        
        Scans parked segments and grades them against the new state.
        Segments above the relevance threshold are re-activated.
        
        Args:
            state: New state/context to match against (uses current_state if None)
            min_relevance: Minimum usefulness score to recall (0-1)
            
        Returns:
            List of recalled segments
        """
        if state:
            self.current_state = state
        
        if min_relevance is None:
            min_relevance = self.config.recall_relevance_threshold
        
        recalled = []
        
        for seg_id, seg in list(self.bank.items()):
            # Grade usefulness against new state
            usefulness = self.grader.grade_usefulness(self.current_state, seg.content)
            
            if usefulness >= min_relevance:
                # Re-activate segment
                seg.status = SegmentStatus.ACTIVE
                seg.usefulness_score = usefulness
                seg.calculate_priority()
                
                # Move from bank to active store
                del self.bank[seg_id]
                self.store.add_segment(seg)
                recalled.append(seg)
        
        if recalled:
            self.store.log_action("recall", {
                "recalled": len(recalled),
                "state": self.current_state
            })
        
        return recalled
    
    def tag_segment(self, seg_id: str, tags: List[str],
                     task_id: Optional[str] = None,
                     task_description: Optional[str] = None):
        """Tag a segment for task-specific recall.
        
        Tags allow specific segments to be recalled when the matching
        task is active.
        
        Args:
            seg_id: Segment ID to tag
            tags: List of tag strings
            task_id: Optional task identifier
            task_description: Optional task description for context
        """
        seg = self.store.get_segment(seg_id) or self.bank.get(seg_id)
        if not seg:
            return
        
        seg.metadata["tags"] = tags
        if task_id:
            seg.metadata["task_id"] = task_id
        if task_description:
            seg.metadata["task_description"] = task_description
        
        if seg_id in self.store.segments:
            self.store.add_segment(seg)
    
    def recall_by_task(self, task_id: str) -> List[Segment]:
        """Recall all segments tagged with a specific task ID.
        
        Args:
            task_id: Task identifier to match
            
        Returns:
            List of recalled segments
        """
        recalled = []
        
        for seg_id, seg in list(self.bank.items()):
            if seg.metadata.get("task_id") == task_id:
                seg.status = SegmentStatus.ACTIVE
                seg.calculate_priority()
                del self.bank[seg_id]
                self.store.add_segment(seg)
                recalled.append(seg)
        
        if recalled:
            self.store.log_action("recall_by_task", {
                "task_id": task_id,
                "recalled": len(recalled)
            })
        
        return recalled
    
    def bank_context(self, max_tokens: int = 4096,
                      state: Optional[str] = None) -> List[Segment]:
        """Proactively bank currently irrelevant segments.
        
        Grades all active segments and banks those below the relevance
        threshold. This is useful before a state change to preserve
        context that might be needed later.
        
        Args:
            max_tokens: Token budget for context
            state: State to grade against
            
        Returns:
            List of banked segments
        """
        if state:
            self.current_state = state
        
        active = self.store.get_active_segments()
        banked = []
        
        for seg in active:
            usefulness = self.grader.grade_usefulness(self.current_state, seg.content)
            
            if usefulness < self.config.bank_irrelevance_threshold:  # Below relevance threshold
                seg.status = SegmentStatus.BANKED
                seg.metadata["banked_reason"] = "irrelevant_to_state"
                self.bank[seg.id] = seg
                banked.append(seg)
                self.store.add_segment(seg)
        
        if banked:
            self.store.log_action("bank_context", {
                "banked": len(banked),
                "state": self.current_state
            })
        
        return banked
    
    def get_bank(self) -> List[Segment]:
        """Get all banked (parked) segments.
        
        Returns:
            List of banked segments
        """
        return list(self.bank.values())
    
    def get_bank_summary(self) -> List[Dict[str, Any]]:
        """Get summary of banked segments.
        
        Returns:
            List of bank summaries with id, content preview, and tags
        """
        return [
            {
                "id": seg.id,
                "content": seg.content[:100],
                "type": seg.type.value,
                "tags": seg.metadata.get("tags", []),
                "task_id": seg.metadata.get("task_id"),
                "banked_at": seg.metadata.get("banked_at")
            }
            for seg in self.bank.values()
        ]
    
    def correct(self, corrections: List[Dict[str, str]]) -> int:
        """Apply corrections to segments.
        
        Args:
            corrections: List of {segment_id: corrected_content}
            
        Returns:
            Number of corrections applied
        """
        applied = 0
        for corr in corrections:
            seg_id = corr.get("id", "")
            new_content = corr.get("content", "")
            
            seg = self.store.get_segment(seg_id)
            if seg and seg.status == SegmentStatus.ACTIVE:
                seg.content = new_content
                seg.status = SegmentStatus.ALTERED
                seg.calculate_hash()
                self.store.add_segment(seg)
                applied += 1
        
        self.store.log_action("correct", {"applied": applied})
        return applied
    
    def fact_check(self,
                   claim: str,
                   state: Optional[str] = None,
                   threshold: float = 0.7) -> Dict[str, Any]:
        """Check a claim against SystemOne for truthfulness.
        
        Returns a gray-scale truthfulness score (0.0 = definitely false,
        1.0 = definitely true) and a recommendation.
        
        Args:
            claim: The statement to check
            state: Context/state for the check
            threshold: Score above which claim is considered true
            
        Returns:
            Dict with score, verdict, and recommendation
        """
        if state:
            self.current_state = state
        
        score = self.grader.grade_truthfulness(self.current_state, claim)
        
        # Determine verdict based on score
        if score >= threshold:
            verdict = "verified"
            recommendation = "keep"
        elif score <= (1.0 - threshold):
            verdict = "false"
            recommendation = "remove_or_correct"
        else:
            verdict = "contested"
            recommendation = "bank_or_investigate"
        
        return {
            "claim": claim,
            "score": score,
            "verdict": verdict,
            "recommendation": recommendation,
            "confidence": abs(score - 0.5) * 2,  # 0-1 confidence
            "checked_at": time.time()
        }
    
    def mark_contested(self,
                       seg_id: str,
                       reason: str,
                       confidence: float = 0.5,
                       evidence: Optional[List[str]] = None):
        """Mark a segment as contested (disputed).
        
        Used when something might be wrong but isn't confirmed false.
        The segment stays in context but with reduced priority.
        
        Args:
            seg_id: Segment ID to mark
            reason: Why it's contested
            confidence: How confident we are it's wrong (0-1)
                       0.0 = definitely wrong, 1.0 = definitely right
            evidence: List of evidence strings supporting the contest
        """
        seg = self.store.get_segment(seg_id)
        if not seg:
            return
        
        seg.mark_contested(reason, confidence)
        if evidence:
            seg.metadata["contested_evidence"] = evidence
        
        self.store.add_segment(seg)
        self.store.log_action("mark_contested", {
            "seg_id": seg_id,
            "reason": reason,
            "confidence": confidence
        })
    
    def mark_false(self,
                   seg_id: str,
                   reason: str,
                   corrected_content: Optional[str] = None,
                   evidence: Optional[List[str]] = None):
        """Mark a segment as confirmed false (anti-fact).
        
        The segment is excluded from active context and marked as false.
        A corrected version can be stored for reference.
        
        Args:
            seg_id: Segment ID to mark
            reason: Why it's false
            corrected_content: Optional corrected version
            evidence: List of evidence strings
        """
        seg = self.store.get_segment(seg_id)
        if not seg:
            return
        
        seg.mark_false(reason, corrected_content)
        if evidence:
            seg.metadata["false_evidence"] = evidence
        
        self.store.add_segment(seg)
        self.store.log_action("mark_false", {
            "seg_id": seg_id,
            "reason": reason,
            "has_correction": corrected_content is not None
        })
    
    def mark_verified(self, seg_id: str, verified_by: str = "systemone"):
        """Mark a segment as verified correct.
        
        Args:
            seg_id: Segment ID to verify
            verified_by: Who verified it (default: systemone)
        """
        seg = self.store.get_segment(seg_id)
        if not seg:
            return
        
        seg.mark_verified(verified_by)
        self.store.add_segment(seg)
        self.store.log_action("mark_verified", {
            "seg_id": seg_id,
            "verified_by": verified_by
        })
    
    def get_contested(self) -> List[Segment]:
        """Get all contested segments.
        
        Returns:
            List of contested segments
        """
        return [s for s in self.store.get_all_segments() 
                if s.status == SegmentStatus.CONTESTED]
    
    def get_false_segments(self) -> List[Segment]:
        """Get all segments marked as false.
        
        Returns:
            List of false segments
        """
        return [s for s in self.store.get_all_segments() 
                if s.status == SegmentStatus.FALSE]
    
    def get_truthfulness_report(self) -> List[Dict[str, Any]]:
        """Get a report of all segments sorted by truthfulness score.
        
        Useful for auditing what has been verified vs contested vs false.
        
        Returns:
            List of segment summaries with truthfulness scores
        """
        all_segs = self.store.get_all_segments()
        report = []
        
        for seg in sorted(all_segs, key=lambda s: s.truthfulness_score, reverse=True):
            entry = {
                "id": seg.id,
                "content": seg.content[:100],
                "type": seg.type.value,
                "status": seg.status.value,
                "truthfulness": seg.truthfulness_score,
                "usefulness": seg.usefulness_score,
            }
            
            # Add metadata for contested/false
            if seg.metadata.get("contested_reason"):
                entry["contested_reason"] = seg.metadata["contested_reason"]
                entry["contested_confidence"] = seg.metadata.get("contested_confidence", 0.5)
            
            if seg.metadata.get("false_reason"):
                entry["false_reason"] = seg.metadata["false_reason"]
                entry["corrected_content"] = seg.metadata.get("corrected_content")
            
            report.append(entry)
        
        return report
    
    def audit_facts(self, state: Optional[str] = None,
                    contested_threshold: float = 0.3,
                    false_threshold: float = 0.1) -> Dict[str, Any]:
        """Audit all segments for factual accuracy using SystemOne.
        
        Grades each segment and automatically marks:
        - Segments below false_threshold as FALSE
        - Segments between false_threshold and contested_threshold as CONTESTED
        - Segments above contested_threshold as verified
        
        Args:
            state: State/context for the audit
            contested_threshold: Score below which segment is contested
            false_threshold: Score below which segment is marked false
            
        Returns:
            Audit report with counts and actions taken
        """
        if state:
            self.current_state = state
        
        active = self.store.get_active_segments()
        results = {"verified": 0, "contested": 0, "false": 0, "actions": []}
        
        for seg in active:
            score = self.grader.grade_truthfulness(self.current_state, seg.content)
            seg.truthfulness_score = score
            
            if score < false_threshold:
                seg.mark_false(f"SystemOne audit: score {score:.2f} below threshold",
                              None)
                self.store.add_segment(seg)
                results["false"] += 1
                results["actions"].append({
                    "seg_id": seg.id,
                    "action": "marked_false",
                    "score": score
                })
            elif score < contested_threshold:
                seg.mark_contested(
                    f"SystemOne audit: score {score:.2f} in gray zone",
                    confidence=score  # Lower confidence = more likely wrong
                )
                self.store.add_segment(seg)
                results["contested"] += 1
                results["actions"].append({
                    "seg_id": seg.id,
                    "action": "marked_contested",
                    "score": score
                })
            else:
                seg.mark_verified("systemone_audit")
                self.store.add_segment(seg)
                results["verified"] += 1
        
        self.store.log_action("audit_facts", {
            "total": len(active),
            "verified": results["verified"],
            "contested": results["contested"],
            "false": results["false"]
        })
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get context statistics.
        
        Returns:
            Stats dict with counts and average scores
        """
        active = self.store.get_active_segments()
        banked = list(self.bank.values())
        
        if active:
            avg_truthfulness = sum(s.truthfulness_score for s in active) / len(active)
            avg_usefulness = sum(s.usefulness_score for s in active) / len(active)
        else:
            avg_truthfulness = 0.0
            avg_usefulness = 0.0
        
        return {
            "total_segments": len(self.store.get_all_segments()),
            "active": len(active),
            "banked": len(banked),
            "avg_truthfulness": avg_truthfulness,
            "avg_usefulness": avg_usefulness,
            "systemone_available": self.grader.available
        }
