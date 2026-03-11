"""NetworkX-based knowledge graph — central data store."""
from __future__ import annotations
import json, os
from typing import Dict, List, Any
from pathlib import Path
import networkx as nx
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models import ModuleNode, DatasetNode, FunctionNode, TransformationNode


class KnowledgeGraph:
    def __init__(self):
        self.module_graph: nx.DiGraph = nx.DiGraph()
        self.lineage_graph: nx.DiGraph = nx.DiGraph()
        self.modules: Dict[str, ModuleNode] = {}
        self.datasets: Dict[str, DatasetNode] = {}
        self.functions: Dict[str, FunctionNode] = {}
        self.transformations: Dict[str, TransformationNode] = {}

    def add_module(self, node: ModuleNode):
        self.modules[node.path] = node
        attrs = {k: v for k, v in node.to_dict().items() if not isinstance(v, (list, dict))}
        self.module_graph.add_node(node.path, **attrs)

    def add_import_edge(self, source: str, target: str, weight: int = 1):
        if source in self.module_graph and target in self.module_graph:
            if self.module_graph.has_edge(source, target):
                self.module_graph[source][target]['weight'] += weight
            else:
                self.module_graph.add_edge(source, target, edge_type="IMPORTS", weight=weight)

    def add_function(self, node: FunctionNode):
        self.functions[node.qualified_name] = node

    def add_dataset(self, node: DatasetNode):
        self.datasets[node.name] = node
        attrs = {k: v for k, v in node.to_dict().items() if not isinstance(v, (list, dict))}
        self.lineage_graph.add_node(node.name, **attrs)

    def add_transformation(self, node: TransformationNode):
        self.transformations[node.name] = node
        attrs = {k: v for k, v in node.to_dict().items() if not isinstance(v, (list, dict))}
        self.lineage_graph.add_node(node.name, **attrs)
        for src in node.source_datasets:
            if src not in self.datasets:
                self.add_dataset(DatasetNode(name=src))
            self.lineage_graph.add_edge(src, node.name, edge_type="CONSUMES")
        for tgt in node.target_datasets:
            if tgt not in self.datasets:
                self.add_dataset(DatasetNode(name=tgt))
            self.lineage_graph.add_edge(node.name, tgt, edge_type="PRODUCES")

    def compute_pagerank(self) -> Dict[str, float]:
        if len(self.module_graph) == 0:
            return {}
        try:
            return nx.pagerank(self.module_graph, weight='weight')
        except Exception:
            return {n: 1.0/len(self.module_graph) for n in self.module_graph.nodes}

    def find_circular_dependencies(self) -> List[List[str]]:
        try:
            return [sorted(s) for s in nx.strongly_connected_components(self.module_graph) if len(s) > 1]
        except Exception:
            return []

    def blast_radius(self, node_path: str, max_depth: int = 10) -> Dict[str, Any]:
        result = {"source": node_path, "downstream_modules": [], "downstream_datasets": [],
                  "total_affected": 0, "depth_map": {}}
        if node_path in self.module_graph:
            visited = {}
            queue = [(node_path, 0)]
            while queue:
                current, depth = queue.pop(0)
                if current in visited or depth > max_depth:
                    continue
                visited[current] = depth
                result["depth_map"][current] = depth
                # predecessors = modules that import 'current' — they break if 'current' changes
                for s in self.module_graph.predecessors(current):
                    if s not in visited:
                        queue.append((s, depth + 1))
            result["downstream_modules"] = [n for n in visited if n != node_path]
            result["total_affected"] = len(result["downstream_modules"])
        key = os.path.basename(node_path).replace(".py", "")
        for ds in self.datasets:
            if key.lower() in ds.lower() or ds.lower() in key.lower():
                try:
                    result["downstream_datasets"] = list(nx.descendants(self.lineage_graph, ds))
                except Exception:
                    pass
        return result

    def trace_lineage(self, dataset_name: str, direction: str = "upstream") -> Dict[str, Any]:
        result = {"dataset": dataset_name, "direction": direction, "nodes": [], "edges": [], "found": False}
        if dataset_name not in self.lineage_graph:
            for name in self.datasets:
                if dataset_name.lower() in name.lower():
                    dataset_name = name; break
        if dataset_name not in self.lineage_graph:
            return result
        result["found"] = True
        try:
            if direction == "upstream":
                nodes = nx.ancestors(self.lineage_graph, dataset_name)
            else:
                nodes = nx.descendants(self.lineage_graph, dataset_name)
            sub = self.lineage_graph.subgraph(nodes | {dataset_name})
            result["nodes"] = list(nodes)
            result["edges"] = [{"source": u, "target": v, "type": d.get("edge_type","?")}
                               for u, v, d in sub.edges(data=True)]
        except Exception as e:
            result["error"] = str(e)
        return result

    def find_sources(self) -> List[str]:
        return [n for n in self.lineage_graph.nodes
                if self.lineage_graph.in_degree(n) == 0 and n in self.datasets]

    def find_sinks(self) -> List[str]:
        return [n for n in self.lineage_graph.nodes
                if self.lineage_graph.out_degree(n) == 0 and n in self.datasets]

    def save_module_graph(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(nx.node_link_data(self.module_graph), f, indent=2, default=str)

    def save_lineage_graph(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        output = {
            "graph": nx.node_link_data(self.lineage_graph),
            "datasets": {k: v.to_dict() for k, v in self.datasets.items()},
            "transformations": {k: v.to_dict() for k, v in self.transformations.items()},
            "sources": self.find_sources(),
            "sinks": self.find_sinks(),
        }
        with open(path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
