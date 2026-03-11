"""CartographyResult + AnalysisTrace containers."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any
import json
from .nodes import ModuleNode, DatasetNode, FunctionNode, TransformationNode
from .edges import GraphEdge


@dataclass
class AnalysisTrace:
    """Single audit-log entry written to cartography_trace.jsonl."""
    timestamp: str
    agent: str
    action: str
    target: str
    result_summary: str
    evidence_source: str = ""   # "static_analysis" | "llm_inference" | "git_log"
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        return json.dumps({"timestamp":self.timestamp,"agent":self.agent,
            "action":self.action,"target":self.target,
            "result_summary":self.result_summary,"evidence_source":self.evidence_source,
            "confidence":self.confidence,"metadata":self.metadata})


@dataclass
class CartographyResult:
    """Top-level container for all cartography outputs."""
    repo_path: str
    repo_name: str
    analysis_timestamp: str
    module_nodes: Dict[str, ModuleNode] = field(default_factory=dict)
    dataset_nodes: Dict[str, DatasetNode] = field(default_factory=dict)
    function_nodes: Dict[str, FunctionNode] = field(default_factory=dict)
    transformation_nodes: Dict[str, TransformationNode] = field(default_factory=dict)
    edges: List[GraphEdge] = field(default_factory=list)
    domain_clusters: Dict[str, List[str]] = field(default_factory=dict)
    day_one_answers: Dict[str, str] = field(default_factory=dict)
    high_velocity_files: List[Any] = field(default_factory=list)
    circular_dependencies: List[List[str]] = field(default_factory=list)
    pagerank_scores: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"repo_path":self.repo_path,"repo_name":self.repo_name,
            "analysis_timestamp":self.analysis_timestamp,
            "module_nodes":{k:v.to_dict() for k,v in self.module_nodes.items()},
            "dataset_nodes":{k:v.to_dict() for k,v in self.dataset_nodes.items()},
            "function_nodes":{k:v.to_dict() for k,v in self.function_nodes.items()},
            "transformation_nodes":{k:v.to_dict() for k,v in self.transformation_nodes.items()},
            "edges":[e.to_dict() for e in self.edges],
            "domain_clusters":self.domain_clusters,"day_one_answers":self.day_one_answers,
            "high_velocity_files":self.high_velocity_files,
            "circular_dependencies":self.circular_dependencies,
            "pagerank_scores":self.pagerank_scores,"errors":self.errors}
