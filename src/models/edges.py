"""Edge type definitions with runtime validation."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict
from enum import Enum
from .nodes import ValidationError


class EdgeType(str, Enum):
    IMPORTS    = "IMPORTS"
    PRODUCES   = "PRODUCES"
    CONSUMES   = "CONSUMES"
    CALLS      = "CALLS"
    CONFIGURES = "CONFIGURES"
    CONTAINS   = "CONTAINS"

    @classmethod
    def coerce(cls, v):
        if isinstance(v, cls): return v
        try: return cls(str(v).upper())
        except ValueError: raise ValidationError(f"Invalid EdgeType: {v!r}")


@dataclass
class GraphEdge:
    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.source: raise ValidationError("GraphEdge.source must be non-empty")
        if not self.target: raise ValidationError("GraphEdge.target must be non-empty")
        self.edge_type = EdgeType.coerce(self.edge_type)
        self.weight = float(self.weight if self.weight is not None else 1.0)
        if self.weight <= 0: raise ValidationError(f"GraphEdge.weight must be > 0, got {self.weight}")
        self.metadata = self.metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {"source":self.source,"target":self.target,
            "edge_type":self.edge_type.value,"weight":self.weight,"metadata":self.metadata}
