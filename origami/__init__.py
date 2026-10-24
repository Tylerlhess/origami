"""origami — Context/memory management service using SystemOne."""

from .segment import Segment, SegmentType, SegmentStatus
from .dissector import SegmentDissector
from .grader import SystemOneGrader
from .manager import ContextManager
from .store import ContextStore
from .config import OrigamiConfig

__all__ = [
    "Segment", "SegmentType", "SegmentStatus",
    "SegmentDissector", "SystemOneGrader",
    "ContextManager", "ContextStore",
    "OrigamiConfig"
]

__version__ = "0.1.0"
