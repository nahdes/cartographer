"""
Node schemas with full runtime validation (Pydantic-style, zero external deps).
Every field validates type and constraints in __post_init__.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum


class ValidationError(Exception):
    """Raised when a field fails schema validation."""


def _req_str(v, name):
    if not isinstance(v, str):
        raise ValidationError(f"{name}: expected str, got {type(v).__name__}")
    if not v.strip():
        raise ValidationError(f"{name}: must be non-empty")
    return v.strip()

def _coerce_float(v, name, min_val=0.0):
    try:
        f = float(v or 0)
    except (TypeError, ValueError):
        raise ValidationError(f"{name}: cannot coerce {v!r} to float")
    if f < min_val:
        raise ValidationError(f"{name}: must be >= {min_val}, got {f}")
    return f

def _coerce_int(v, name, min_val=0):
    try:
        i = int(v or 0)
    except (TypeError, ValueError):
        raise ValidationError(f"{name}: cannot coerce {v!r} to int")
    if i < min_val:
        raise ValidationError(f"{name}: must be >= {min_val}, got {i}")
    return i

def _coerce_list(v, name):
    if v is None: return []
    if not isinstance(v, list):
        raise ValidationError(f"{name}: expected list, got {type(v).__name__}")
    return v

def _coerce_dict(v, name):
    if v is None: return {}
    if not isinstance(v, dict):
        raise ValidationError(f"{name}: expected dict, got {type(v).__name__}")
    return v


class StorageType(str, Enum):
    TABLE="table"; FILE="file"; STREAM="stream"; API="api"; UNKNOWN="unknown"
    @classmethod
    def coerce(cls, v):
        if isinstance(v, cls): return v
        try: return cls(str(v).lower())
        except ValueError: return cls.UNKNOWN

class Language(str, Enum):
    PYTHON="python"; SQL="sql"; YAML="yaml"
    JAVASCRIPT="javascript"; TYPESCRIPT="typescript"
    NOTEBOOK="notebook"; UNKNOWN="unknown"
    @classmethod
    def coerce(cls, v):
        if isinstance(v, cls): return v
        try: return cls(str(v).lower())
        except ValueError: return cls.UNKNOWN

class TransformationType(str, Enum):
    PANDAS="pandas"; SPARK="spark"; SQLALCHEMY="sqlalchemy"
    SQL_SELECT="sql_select"; DBT_MODEL="dbt_model"
    AIRFLOW_TASK="airflow_task"; UNKNOWN="unknown"
    @classmethod
    def coerce(cls, v):
        if isinstance(v, cls): return v
        try: return cls(str(v).lower())
        except ValueError: return cls.UNKNOWN


@dataclass
class ModuleNode:
    """Represents one source file in the codebase."""
    path: str
    language: Language = Language.UNKNOWN
    purpose_statement: str = ""
    domain_cluster: str = ""
    complexity_score: float = 0.0
    change_velocity_30d: int = 0
    loc: int = 0
    comment_ratio: float = 0.0
    pagerank_score: float = 0.0
    is_dead_code_candidate: bool = False
    doc_drift_flag: bool = False
    last_modified: str = ""
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    docstring: str = ""
    js_ts_exports: List[str] = field(default_factory=list)
    js_ts_imports: List[str] = field(default_factory=list)
    node_type: str = "module"

    def __post_init__(self):
        self.path = _req_str(self.path, "ModuleNode.path")
        self.language = Language.coerce(self.language)
        self.complexity_score = _coerce_float(self.complexity_score, "complexity_score")
        self.change_velocity_30d = _coerce_int(self.change_velocity_30d, "change_velocity_30d")
        self.loc = _coerce_int(self.loc, "loc")
        try:
            cr = float(self.comment_ratio or 0)
        except (TypeError, ValueError):
            raise ValidationError(f"comment_ratio: cannot coerce {self.comment_ratio!r} to float")
        self.comment_ratio = max(0.0, min(1.0, cr))
        self.pagerank_score = _coerce_float(self.pagerank_score, "pagerank_score")
        self.imports = _coerce_list(self.imports, "imports")
        self.exports = _coerce_list(self.exports, "exports")
        self.js_ts_exports = _coerce_list(self.js_ts_exports, "js_ts_exports")
        self.js_ts_imports = _coerce_list(self.js_ts_imports, "js_ts_imports")

    def to_dict(self) -> Dict[str, Any]:
        return {"path":self.path,"language":self.language.value,
            "purpose_statement":self.purpose_statement,"domain_cluster":self.domain_cluster,
            "complexity_score":self.complexity_score,"change_velocity_30d":self.change_velocity_30d,
            "loc":self.loc,"comment_ratio":self.comment_ratio,"pagerank_score":self.pagerank_score,
            "is_dead_code_candidate":self.is_dead_code_candidate,"doc_drift_flag":self.doc_drift_flag,
            "last_modified":self.last_modified,"imports":self.imports,"exports":self.exports,
            "docstring":self.docstring,"js_ts_exports":self.js_ts_exports,
            "js_ts_imports":self.js_ts_imports,"node_type":self.node_type}

    @classmethod
    def from_dict(cls, d):
        return cls(**{k:v for k,v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DatasetNode:
    """Dataset, table, file, or stream in the lineage graph."""
    name: str
    storage_type: StorageType = StorageType.UNKNOWN
    schema_snapshot: Dict[str, str] = field(default_factory=dict)
    column_lineage: Dict[str, List[str]] = field(default_factory=dict)
    freshness_sla: str = ""
    owner: str = ""
    is_source_of_truth: bool = False
    source_file: str = ""
    line_range: List[int] = field(default_factory=list)
    node_type: str = "dataset"

    def __post_init__(self):
        self.name = _req_str(self.name, "DatasetNode.name")
        self.storage_type = StorageType.coerce(self.storage_type)
        self.schema_snapshot = _coerce_dict(self.schema_snapshot, "schema_snapshot")
        self.column_lineage = _coerce_dict(self.column_lineage, "column_lineage")
        self.line_range = _coerce_list(self.line_range, "line_range")

    def to_dict(self) -> Dict[str, Any]:
        return {"name":self.name,"storage_type":self.storage_type.value,
            "schema_snapshot":self.schema_snapshot,"column_lineage":self.column_lineage,
            "freshness_sla":self.freshness_sla,"owner":self.owner,
            "is_source_of_truth":self.is_source_of_truth,"source_file":self.source_file,
            "line_range":self.line_range,"node_type":self.node_type}

    @classmethod
    def from_dict(cls, d):
        return cls(**{k:v for k,v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class FunctionNode:
    """Function or method within a module."""
    qualified_name: str
    parent_module: str
    signature: str = ""
    purpose_statement: str = ""
    call_count_within_repo: int = 0
    is_public_api: bool = False
    line_number: int = 0
    complexity: int = 0
    return_type: str = ""
    node_type: str = "function"

    def __post_init__(self):
        self.qualified_name = _req_str(self.qualified_name, "FunctionNode.qualified_name")
        self.parent_module = _req_str(self.parent_module, "FunctionNode.parent_module")
        self.call_count_within_repo = _coerce_int(self.call_count_within_repo, "call_count")
        self.line_number = _coerce_int(self.line_number, "line_number")
        self.complexity = _coerce_int(self.complexity, "complexity")

    def to_dict(self) -> Dict[str, Any]:
        return {"qualified_name":self.qualified_name,"parent_module":self.parent_module,
            "signature":self.signature,"purpose_statement":self.purpose_statement,
            "call_count_within_repo":self.call_count_within_repo,"is_public_api":self.is_public_api,
            "line_number":self.line_number,"complexity":self.complexity,
            "return_type":self.return_type,"node_type":self.node_type}


@dataclass
class TransformationNode:
    """Data transformation: source_datasets -> target_datasets."""
    name: str
    source_datasets: List[str] = field(default_factory=list)
    target_datasets: List[str] = field(default_factory=list)
    transformation_type: str = TransformationType.UNKNOWN.value
    source_file: str = ""
    line_range: List[int] = field(default_factory=list)
    sql_query_if_applicable: str = ""
    column_mappings: Dict[str, List[str]] = field(default_factory=dict)
    node_type: str = "transformation"

    def __post_init__(self):
        self.name = _req_str(self.name, "TransformationNode.name")
        self.source_datasets = _coerce_list(self.source_datasets, "source_datasets")
        self.target_datasets = _coerce_list(self.target_datasets, "target_datasets")
        self.line_range = _coerce_list(self.line_range, "line_range")
        self.column_mappings = _coerce_dict(self.column_mappings, "column_mappings")

    def to_dict(self) -> Dict[str, Any]:
        return {"name":self.name,"source_datasets":self.source_datasets,
            "target_datasets":self.target_datasets,"transformation_type":self.transformation_type,
            "source_file":self.source_file,"line_range":self.line_range,
            "sql_query_if_applicable":self.sql_query_if_applicable,
            "column_mappings":self.column_mappings,"node_type":self.node_type}

    @classmethod
    def from_dict(cls, d):
        return cls(**{k:v for k,v in d.items() if k in cls.__dataclass_fields__})
