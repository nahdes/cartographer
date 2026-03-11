"""
src/models/pydantic_schemas.py — Optional Pydantic v2 schemas.

When `pydantic` is installed these provide:
  • Full runtime validation with clear error messages
  • JSON schema generation  (e.g. for OpenAPI / tool-use schemas)
  • .model_validate(dict) factory that coerces & validates in one call
  • .model_dump() consistent serialisation

When pydantic is NOT installed, thin shim classes fall back to the
dataclass models in nodes.py/edges.py — zero import errors, zero
behaviour change.

Usage:
    from src.models.pydantic_schemas import (
        ModuleNodeSchema, DatasetNodeSchema, TransformationNodeSchema,
        FunctionNodeSchema, GraphEdgeSchema, PYDANTIC_AVAILABLE,
    )
    if PYDANTIC_AVAILABLE:
        schema = ModuleNodeSchema.model_validate(module_node.to_dict())
        print(schema.model_json_schema())
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

# ── Try to import Pydantic v2 ─────────────────────────────────────────────────
try:
    from pydantic import (
        BaseModel, Field, field_validator, model_validator,
        ConfigDict, ValidationError as PydanticValidationError,
    )
    import pydantic
    PYDANTIC_AVAILABLE = True
    PYDANTIC_VERSION   = int(pydantic.VERSION.split(".")[0])
    if PYDANTIC_VERSION < 2:
        raise ImportError("Pydantic v2+ required")
except (ImportError, AttributeError):
    PYDANTIC_AVAILABLE = False
    PYDANTIC_VERSION   = 0


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic v2 schemas
# ══════════════════════════════════════════════════════════════════════════════

if PYDANTIC_AVAILABLE:

    class _Base(BaseModel):
        model_config = ConfigDict(
            extra="ignore",          # silently drop unknown fields
            validate_assignment=True, # re-validate on field assignment
            str_strip_whitespace=True,
        )

    # ── ModuleNodeSchema ──────────────────────────────────────────────────────
    class ModuleNodeSchema(_Base):
        path:                   str
        language:               str  = "unknown"
        purpose_statement:      str  = ""
        domain_cluster:         str  = ""
        complexity_score:       float = Field(default=0.0, ge=0.0)
        change_velocity_30d:    int   = Field(default=0,   ge=0)
        loc:                    int   = Field(default=0,   ge=0)
        comment_ratio:          float = Field(default=0.0, ge=0.0, le=1.0)
        pagerank_score:         float = Field(default=0.0, ge=0.0)
        is_dead_code_candidate: bool  = False
        doc_drift_flag:         bool  = False
        last_modified:          str   = ""
        imports:                List[str] = Field(default_factory=list)
        exports:                List[str] = Field(default_factory=list)
        docstring:              str  = ""
        js_ts_exports:          List[str] = Field(default_factory=list)
        js_ts_imports:          List[str] = Field(default_factory=list)
        node_type:              str  = "module"

        @field_validator("path")
        @classmethod
        def path_nonempty(cls, v: str) -> str:
            if not v or not v.strip():
                raise ValueError("path must be non-empty")
            return v.strip()

        @field_validator("comment_ratio", mode="before")
        @classmethod
        def clamp_comment_ratio(cls, v) -> float:
            try:
                f = float(v or 0)
            except (TypeError, ValueError):
                return 0.0
            return max(0.0, min(1.0, f))

        @field_validator("imports", "exports", "js_ts_exports", "js_ts_imports", mode="before")
        @classmethod
        def coerce_list(cls, v) -> List:
            if v is None:
                return []
            return list(v)

    # ── DatasetNodeSchema ─────────────────────────────────────────────────────
    class DatasetNodeSchema(_Base):
        name:               str
        storage_type:       str  = "unknown"
        schema_snapshot:    Dict[str, str]       = Field(default_factory=dict)
        column_lineage:     Dict[str, List[str]] = Field(default_factory=dict)
        freshness_sla:      str  = ""
        owner:              str  = ""
        is_source_of_truth: bool = False
        source_file:        str  = ""
        line_range:         List[int] = Field(default_factory=list)
        node_type:          str  = "dataset"

        @field_validator("name")
        @classmethod
        def name_nonempty(cls, v: str) -> str:
            if not v or not v.strip():
                raise ValueError("name must be non-empty")
            return v.strip()

        @field_validator("schema_snapshot", "column_lineage", mode="before")
        @classmethod
        def coerce_dict(cls, v) -> Dict:
            return v if isinstance(v, dict) else {}

        @field_validator("line_range", mode="before")
        @classmethod
        def coerce_list(cls, v) -> List:
            return v if isinstance(v, list) else []

    # ── FunctionNodeSchema ────────────────────────────────────────────────────
    class FunctionNodeSchema(_Base):
        qualified_name:          str
        parent_module:           str
        signature:               str   = ""
        purpose_statement:       str   = ""
        call_count_within_repo:  int   = Field(default=0, ge=0)
        is_public_api:           bool  = False
        line_number:             int   = Field(default=0, ge=0)
        complexity:              int   = Field(default=0, ge=0)
        return_type:             str   = ""
        node_type:               str   = "function"

        @field_validator("qualified_name", "parent_module")
        @classmethod
        def nonempty(cls, v: str) -> str:
            if not v or not v.strip():
                raise ValueError("field must be non-empty")
            return v.strip()

    # ── TransformationNodeSchema ──────────────────────────────────────────────
    class TransformationNodeSchema(_Base):
        name:                    str
        source_datasets:         List[str] = Field(default_factory=list)
        target_datasets:         List[str] = Field(default_factory=list)
        transformation_type:     str = "unknown"
        source_file:             str = ""
        line_range:              List[int] = Field(default_factory=list)
        sql_query_if_applicable: str = ""
        column_mappings:         Dict[str, List[str]] = Field(default_factory=dict)
        node_type:               str = "transformation"

        @field_validator("name")
        @classmethod
        def name_nonempty(cls, v: str) -> str:
            if not v or not v.strip():
                raise ValueError("name must be non-empty")
            return v.strip()

        @field_validator("source_datasets", "target_datasets", "line_range", mode="before")
        @classmethod
        def coerce_list(cls, v) -> List:
            return v if isinstance(v, list) else ([] if v is None else list(v))

        @field_validator("column_mappings", mode="before")
        @classmethod
        def coerce_dict(cls, v) -> Dict:
            return v if isinstance(v, dict) else {}

    # ── GraphEdgeSchema ───────────────────────────────────────────────────────
    class GraphEdgeSchema(_Base):
        source:    str
        target:    str
        edge_type: str
        weight:    float = Field(default=1.0, gt=0)
        metadata:  Dict[str, Any] = Field(default_factory=dict)

        @field_validator("source", "target")
        @classmethod
        def nonempty(cls, v: str) -> str:
            if not v or not v.strip():
                raise ValueError("source/target must be non-empty")
            return v.strip()

        @field_validator("metadata", mode="before")
        @classmethod
        def coerce_dict(cls, v) -> Dict:
            return v if isinstance(v, dict) else {}

# ══════════════════════════════════════════════════════════════════════════════
# Fallback shims when Pydantic is not installed
# ══════════════════════════════════════════════════════════════════════════════

else:
    # Import dataclass models and expose them under the Schema names.
    # model_validate / model_dump / model_json_schema are added as shims.

    from src.models.nodes import (
        ModuleNode         as _ModuleNode,
        DatasetNode        as _DatasetNode,
        FunctionNode       as _FunctionNode,
        TransformationNode as _TransformationNode,
    )
    from src.models.edges import GraphEdge as _GraphEdge

    def _add_shims(cls, dataclass_cls):
        cls.model_validate = classmethod(
            lambda c, d: dataclass_cls.from_dict(d) if hasattr(dataclass_cls, 'from_dict')
            else dataclass_cls(**{k: v for k, v in d.items()
                                  if k in dataclass_cls.__dataclass_fields__})
        )
        cls.model_dump = lambda self: self.to_dict() if hasattr(self, 'to_dict') else vars(self)
        cls.model_json_schema = classmethod(
            lambda c: {"note": "install pydantic>=2 for full JSON schema"}
        )
        return cls

    ModuleNodeSchema         = _add_shims(_ModuleNode,          _ModuleNode)
    DatasetNodeSchema        = _add_shims(_DatasetNode,         _DatasetNode)
    FunctionNodeSchema       = _add_shims(_FunctionNode,        _FunctionNode)
    TransformationNodeSchema = _add_shims(_TransformationNode,  _TransformationNode)
    GraphEdgeSchema          = _add_shims(_GraphEdge,           _GraphEdge)
    PydanticValidationError  = None


__all__ = [
    "PYDANTIC_AVAILABLE", "PYDANTIC_VERSION",
    "ModuleNodeSchema", "DatasetNodeSchema", "FunctionNodeSchema",
    "TransformationNodeSchema", "GraphEdgeSchema",
]
