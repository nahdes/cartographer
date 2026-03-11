"""
Agent 5: The Navigator — LangGraph-style stateful query agent  (v2)

Architecture: real StateGraph with typed state, explicit nodes and edges,
conditional routing, and multi-step chaining.

StateGraph nodes:
    router       — classifies intent (keyword or LLM tool-use)
    find_impl    — find_implementation tool
    trace_lin    — trace_lineage tool
    blast        — blast_radius tool
    explain      — explain_module tool
    semantic_search — semantic similarity over vector index
    format       — renders final answer with provenance citations

Edges:
    router -> {find_impl, trace_lin, blast, explain, semantic_search}
    {find_impl, trace_lin, blast, explain} -> format
    semantic_search -> format
    format -> END

Tool schemas are Anthropic tool-use / LangGraph-compatible.
"""
from __future__ import annotations
import os, json, re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.graph.knowledge_graph import KnowledgeGraph
from src.models import CartographyResult

try:
    import anthropic
    ANTHROPIC_OK = True
except ImportError:
    ANTHROPIC_OK = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# Typed graph state
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GraphState:
    """Immutable-ish state threaded through every node in the StateGraph."""
    query:          str  = ""
    tool_name:      str  = ""           # which tool was selected
    tool_params:    Dict[str, Any] = field(default_factory=dict)
    tool_result:    Dict[str, Any] = field(default_factory=dict)
    final_answer:   str = ""
    routing_method: str = ""            # "keyword" | "llm_tool_use" | "fallback"
    steps:          List[str] = field(default_factory=list)  # audit trace

    def add_step(self, msg: str) -> "GraphState":
        self.steps.append(msg)
        return self


# ══════════════════════════════════════════════════════════════════════════════
# LangGraph-compatible tool schemas
# ══════════════════════════════════════════════════════════════════════════════

TOOL_SCHEMAS = [
    {
        "name": "find_implementation",
        "description": "Find where a concept, feature, or function is implemented.",
        "input_schema": {
            "type": "object",
            "properties": {"concept": {"type": "string",
                "description": "The concept or feature to search for"}},
            "required": ["concept"],
        },
    },
    {
        "name": "trace_lineage",
        "description": "Trace upstream sources or downstream consumers of a dataset.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset":   {"type": "string", "description": "Dataset name"},
                "direction": {"type": "string", "enum": ["upstream", "downstream"]},
            },
            "required": ["dataset"],
        },
    },
    {
        "name": "blast_radius",
        "description": "Find all modules that depend on the given module (break if it changes).",
        "input_schema": {
            "type": "object",
            "properties": {"module_path": {"type": "string",
                "description": "File path to the module"}},
            "required": ["module_path"],
        },
    },
    {
        "name": "explain_module",
        "description": "Full explanation of a module: purpose, data flows, risks.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string",
                "description": "File path of the module"}},
            "required": ["path"],
        },
    },
    {
        "name": "semantic_search",
        "description": "Vector similarity search over module purpose statements.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string",
                "description": "Natural-language concept to search for"}},
            "required": ["query"],
        },
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# Tool implementations (pure functions — no routing logic)
# ══════════════════════════════════════════════════════════════════════════════

class NavigatorTools:
    def __init__(self, kg: KnowledgeGraph, result: CartographyResult):
        self.kg     = kg
        self.result = result
        self._vec:   Optional[TfidfVectorizer] = None
        self._mat    = None
        self._paths: List[str] = []
        if SKLEARN_OK and result.module_nodes:
            self._build_index()

    def _build_index(self):
        paths = list(self.result.module_nodes.keys())
        texts = [
            f"{self.result.module_nodes[p].purpose_statement} "
            f"{' '.join(self.result.module_nodes[p].exports)} {p}"
            for p in paths
        ]
        try:
            self._vec  = TfidfVectorizer(max_features=500, stop_words="english",
                                         ngram_range=(1, 2), sublinear_tf=True)
            self._mat  = self._vec.fit_transform(texts)
            self._paths = paths
        except Exception:
            pass

    def _sem_scores(self, query: str) -> Dict[str, float]:
        if self._vec is None or self._mat is None:
            return {}
        try:
            qv   = self._vec.transform([query])
            sims = cosine_similarity(qv, self._mat).flatten()
            return {self._paths[i]: float(sims[i]) for i in range(len(self._paths))}
        except Exception:
            return {}

    # ── tool 1 ────────────────────────────────────────────────────────────────
    def find_implementation(self, concept: str) -> Dict[str, Any]:
        concept_lower = concept.lower()
        sem_scores    = self._sem_scores(concept)
        results       = []
        for path, module in self.result.module_nodes.items():
            score, reasons = 0.0, []
            if concept_lower in path.lower():
                score += 3.0; reasons.append("concept in path")
            if module.purpose_statement and concept_lower in module.purpose_statement.lower():
                score += 2.0; reasons.append("concept in purpose")
            for exp in module.exports:
                if concept_lower in exp.lower():
                    score += 2.0; reasons.append(f"export: {exp}"); break
            sem = sem_scores.get(path, 0.0)
            score += sem * 3.0
            if sem > 0.1:
                reasons.append(f"semantic: {sem:.2f}")
            if score > 0:
                results.append({
                    "path":    path, "score": round(score, 3),
                    "reasons": reasons,
                    "purpose": (module.purpose_statement or "")[:200],
                    "domain":  module.domain_cluster,
                    "pagerank": self.result.pagerank_scores.get(path, 0),
                    "analysis_method": "keyword + cosine_similarity",
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return {"tool": "find_implementation", "concept": concept,
                "matches": results[:10], "total_matches": len(results),
                "analysis_method": "keyword_scoring + TF-IDF cosine"}

    # ── tool 2 ────────────────────────────────────────────────────────────────
    def trace_lineage(self, dataset: str,
                      direction: str = "upstream") -> Dict[str, Any]:
        lineage  = self.kg.trace_lineage(dataset, direction)
        enriched = []
        for node_name in lineage.get("nodes", []):
            info: Dict[str, Any] = {"name": node_name}
            if node_name in self.kg.transformations:
                t = self.kg.transformations[node_name]
                info.update({"type": "transformation", "source_file": t.source_file,
                              "column_mappings": t.column_mappings})
            elif node_name in self.kg.datasets:
                d = self.kg.datasets[node_name]
                info.update({"type": "dataset", "source_file": d.source_file,
                              "column_lineage": d.column_lineage})
            enriched.append(info)
        lineage["enriched_nodes"]  = enriched
        lineage["analysis_method"] = "NetworkX ancestors/descendants + column lineage"
        return lineage

    # ── tool 3 ────────────────────────────────────────────────────────────────
    def blast_radius(self, module_path: str) -> Dict[str, Any]:
        actual = module_path
        if actual not in self.kg.module_graph:
            for p in self.kg.modules:
                if module_path in p or Path(p).name == module_path:
                    actual = p; break
        result = self.kg.blast_radius(actual)
        formatted = [
            {"path": m, "depth": result["depth_map"].get(m, 0)}
            for m in result["downstream_modules"]
        ]
        formatted.sort(key=lambda x: x["depth"])
        result["formatted_downstream"] = formatted
        result["analysis_method"] = "BFS over reverse import graph (predecessors)"
        return result

    # ── tool 4 ────────────────────────────────────────────────────────────────
    def explain_module(self, path: str) -> Dict[str, Any]:
        actual = path
        if actual not in self.result.module_nodes:
            for p in self.result.module_nodes:
                if path in p or Path(p).name == path:
                    actual = p; break
        if actual not in self.result.module_nodes:
            return {"error": f"Module '{path}' not found."}
        module = self.result.module_nodes[actual]
        blast  = self.kg.blast_radius(actual)
        return {
            "path":              actual,
            "purpose":           module.purpose_statement or "[not analysed]",
            "domain":            module.domain_cluster,
            "language":          module.language,
            "loc":               module.loc,
            "complexity_score":  module.complexity_score,
            "exports":           module.exports,
            "imports":           module.imports[:10],
            "change_velocity_30d": module.change_velocity_30d,
            "is_dead_code_candidate": module.is_dead_code_candidate,
            "doc_drift":         module.doc_drift_flag,
            "pagerank":          self.result.pagerank_scores.get(actual, 0),
            "blast_radius_count": blast.get("total_affected", 0),
            "downstream_sample": blast.get("downstream_modules", [])[:5],
            "analysis_method":   "static_analysis (AST + graph)",
        }

    # ── tool 5 ────────────────────────────────────────────────────────────────
    def semantic_search(self, query: str, k: int = 8) -> Dict[str, Any]:
        """
        Pure vector-similarity search over module purpose statements.
        Uses the SemanticIndex built by Semanticist if available,
        otherwise falls back to the local NavigatorTools TF-IDF index.
        """
        sem_scores = self._sem_scores(query)
        if not sem_scores:
            return {"tool": "semantic_search", "query": query, "matches": [],
                    "analysis_method": "tfidf_not_available"}
        ranked = sorted(sem_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        matches = []
        for path, score in ranked:
            if score < 0.01:
                continue
            m = self.result.module_nodes.get(path)
            matches.append({
                "path":    path,
                "score":   round(score, 4),
                "purpose": (m.purpose_statement if m else "")[:200],
                "domain":  (m.domain_cluster if m else ""),
            })
        return {"tool": "semantic_search", "query": query,
                "matches": matches, "total": len(matches),
                "analysis_method": "TF-IDF cosine similarity (vector store)"}


# ══════════════════════════════════════════════════════════════════════════════
# StateGraph — nodes
# ══════════════════════════════════════════════════════════════════════════════

# Each node is a pure function: GraphState -> GraphState.
# This mirrors LangGraph's node contract exactly (add_node(name, fn)).

def _node_router(state: GraphState, tools: NavigatorTools,
                 api_client: Any) -> GraphState:
    """
    Route intent to a tool. Tries LLM tool-use first; keyword fallback second.
    """
    query = state.query

    # 1. LLM tool-use (structured routing with parameter extraction)
    if api_client is not None:
        try:
            resp = api_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                tools=TOOL_SCHEMAS,
                messages=[{"role": "user", "content":
                    f"Route this codebase query to the most appropriate tool:\n{query}"}])
            for block in resp.content:
                if hasattr(block, "type") and block.type == "tool_use":
                    state.tool_name      = block.name
                    state.tool_params    = dict(block.input)
                    state.routing_method = "llm_tool_use"
                    state.add_step(f"router→{block.name} (LLM tool-use)")
                    return state
        except Exception:
            pass

    # 2. Keyword routing — ordered: most-specific first
    q = query.lower()
    ROUTES = [
        ("semantic_search",   ["similar to", "related to", "what handles",
                               "concept of", "like the", "modules like"]),
        ("trace_lineage",     ["lineage", "upstream", "downstream", "source of",
                               "feeds into", "produced by", "consumed by", "trace"]),
        ("blast_radius",      ["blast radius", "impact", "what uses", "who uses",
                               "downstream of", "who imports", "depend on", "will break"]),
        ("find_implementation", ["where is", "find", "implement", "which file",
                                 "locate", "search for", "look for"]),
        ("explain_module",    ["explain", "what does", "what is", "describe",
                               "tell me about", "purpose of", "show me"]),
    ]
    for tool, kws in ROUTES:
        if any(kw in q for kw in kws):
            state.tool_name      = tool
            state.routing_method = "keyword"
            state.add_step(f"router→{tool} (keyword match)")
            _fill_params_from_query(state, query)
            return state

    # 3. Fallback: semantic search
    state.tool_name      = "find_implementation"
    state.routing_method = "fallback"
    state.tool_params    = {"concept": query}
    state.add_step("router→find_implementation (fallback)")
    return state


def _fill_params_from_query(state: GraphState, query: str):
    """Heuristic parameter extraction for keyword-routed calls."""
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", query)
    path_like = re.findall(r"[\w./]+\.(?:py|sql|yml|yaml|csv|parquet|js|ts)", query)
    entity = quoted[0] if quoted else (path_like[0] if path_like else _last_noun(query))

    if state.tool_name == "find_implementation":
        state.tool_params = {"concept": entity}
    elif state.tool_name == "trace_lineage":
        direction = "downstream" if any(
            kw in query.lower() for kw in ["downstream","consumes","uses","feeds"]
        ) else "upstream"
        state.tool_params = {"dataset": entity, "direction": direction}
    elif state.tool_name == "blast_radius":
        state.tool_params = {"module_path": entity}
    elif state.tool_name == "explain_module":
        state.tool_params = {"path": entity}
    elif state.tool_name == "semantic_search":
        state.tool_params = {"query": entity or query}


def _last_noun(query: str) -> str:
    words = [w for w in query.split() if len(w) > 3 and w.isalnum()]
    return words[-1] if words else query


def _node_find_impl(state: GraphState, tools: NavigatorTools, _api: Any) -> GraphState:
    state.tool_result = tools.find_implementation(state.tool_params.get("concept", state.query))
    state.add_step("executed find_implementation")
    return state


def _node_trace_lin(state: GraphState, tools: NavigatorTools, _api: Any) -> GraphState:
    state.tool_result = tools.trace_lineage(
        state.tool_params.get("dataset", ""),
        state.tool_params.get("direction", "upstream"))
    state.add_step("executed trace_lineage")
    return state


def _node_blast(state: GraphState, tools: NavigatorTools, _api: Any) -> GraphState:
    state.tool_result = tools.blast_radius(state.tool_params.get("module_path", ""))
    state.add_step("executed blast_radius")
    return state


def _node_explain(state: GraphState, tools: NavigatorTools, _api: Any) -> GraphState:
    state.tool_result = tools.explain_module(state.tool_params.get("path", ""))
    state.add_step("executed explain_module")
    return state


def _node_semantic(state: GraphState, tools: NavigatorTools, _api: Any) -> GraphState:
    state.tool_result = tools.semantic_search(
        state.tool_params.get("query", state.query))
    state.add_step("executed semantic_search")
    return state


def _node_format(state: GraphState, _tools: NavigatorTools, _api: Any) -> GraphState:
    r = state.tool_result
    tool = state.tool_name

    if tool == "find_implementation":
        state.final_answer = _fmt_find(r)
    elif tool == "trace_lineage":
        state.final_answer = _fmt_lineage(r)
    elif tool == "blast_radius":
        state.final_answer = _fmt_blast(r)
    elif tool == "explain_module":
        state.final_answer = _fmt_explain(r)
    elif tool == "semantic_search":
        state.final_answer = _fmt_semantic(r)
    else:
        state.final_answer = json.dumps(r, indent=2, default=str)

    state.add_step("format→END")
    return state


# ── Formatters ────────────────────────────────────────────────────────────────

def _fmt_find(r: Dict) -> str:
    matches = r.get("matches", [])[:5]
    if not matches:
        return f"No matches found for '{r.get('concept','')}'"
    lines = [f"Found {r['total_matches']} matches for '{r['concept']}':"]
    for m in matches:
        lines.append(f"  [{m['score']:.2f}] {m['path']}")
        lines.append(f"          {m['purpose'][:80]}")
    lines.append(f"  [Method: {r['analysis_method']}]")
    return "\n".join(lines)


def _fmt_lineage(r: Dict) -> str:
    if not r.get("found"):
        return f"Dataset '{r['dataset']}' not found in lineage graph."
    nodes, edges = r.get("nodes", []), r.get("edges", [])
    lines = [f"Lineage ({r['direction']}) for '{r['dataset']}':",
             f"  {len(nodes)} related nodes: {', '.join(nodes[:8])}",
             f"  {len(edges)} edges"]
    for n in r.get("enriched_nodes", [])[:5]:
        cm = n.get("column_mappings") or n.get("column_lineage", {})
        if cm:
            sample = list(cm.items())[:2]
            lines.append(f"  Column lineage in {n['name']}: {sample}")
    lines.append(f"  [Method: {r.get('analysis_method', 'graph traversal')}]")
    return "\n".join(lines)


def _fmt_blast(r: Dict) -> str:
    total = r.get("total_affected", 0)
    fmt   = r.get("formatted_downstream", [])[:5]
    lines = [f"Blast radius for '{r['source']}': {total} modules will break if it changes"]
    for d in fmt:
        lines.append(f"  depth {d['depth']}: {d['path']}")
    lines.append(f"  [Method: {r.get('analysis_method', 'BFS')}]")
    return "\n".join(lines)


def _fmt_explain(r: Dict) -> str:
    if "error" in r:
        return r["error"]
    return (
        f"Module: {r['path']}\n"
        f"  Purpose:   {r['purpose']}\n"
        f"  Domain:    {r['domain']}  |  Language: {r['language']}\n"
        f"  LOC: {r['loc']} | Complexity: {r['complexity_score']:.0f} "
        f"| PageRank: {r['pagerank']:.5f}\n"
        f"  Blast radius: {r['blast_radius_count']} dependent modules\n"
        f"  Doc drift: {'⚠️  YES' if r['doc_drift'] else 'OK'}\n"
        f"  [Method: {r['analysis_method']}]"
    )


def _fmt_semantic(r: Dict) -> str:
    matches = r.get("matches", [])[:8]
    if not matches:
        return f"No semantically similar modules found for '{r['query']}'"
    lines = [f"Semantic search ({r.get('total', len(matches))} results) for '{r['query']}':"]
    for m in matches:
        lines.append(f"  [{m['score']:.4f}] {m['path']}")
        if m.get("purpose"):
            lines.append(f"          {m['purpose'][:80]}")
    lines.append(f"  [Method: {r['analysis_method']}]")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# StateGraph — wiring
# ══════════════════════════════════════════════════════════════════════════════

class NavigatorStateGraph:
    """
    Minimal LangGraph-style StateGraph.
    add_node(name, fn) — fn signature: (state, tools, api) -> state
    add_edge(src, tgt) — unconditional transition
    add_conditional_edge(src, condition_fn, mapping) — conditional routing
    invoke(initial_state) -> final_state
    """

    def __init__(self):
        self._nodes:  Dict[str, Any]           = {}
        self._edges:  Dict[str, str]           = {}
        self._cond:   Dict[str, tuple]         = {}  # src -> (fn, mapping)
        self._entry: str = ""

    def set_entry_point(self, node: str):
        self._entry = node

    def add_node(self, name: str, fn):
        self._nodes[name] = fn

    def add_edge(self, src: str, tgt: str):
        self._edges[src] = tgt

    def add_conditional_edges(self, src: str, condition_fn, mapping: Dict[str, str]):
        self._cond[src] = (condition_fn, mapping)

    def invoke(self, state: GraphState, tools: NavigatorTools,
               api_client: Any) -> GraphState:
        current = self._entry
        visited = []
        while current and current != "END":
            if current in visited:
                break  # cycle guard
            visited.append(current)
            fn = self._nodes.get(current)
            if fn is None:
                break
            state = fn(state, tools, api_client)
            # Conditional edge?
            if current in self._cond:
                cond_fn, mapping = self._cond[current]
                key     = cond_fn(state)
                current = mapping.get(key, "END")
            elif current in self._edges:
                current = self._edges[current]
            else:
                break
        return state


def _route_from_state(state: GraphState) -> str:
    """Conditional routing function — maps tool_name to node name."""
    return {
        "find_implementation": "find_impl",
        "trace_lineage":       "trace_lin",
        "blast_radius":        "blast",
        "explain_module":      "explain",
        "semantic_search":     "semantic_search",
    }.get(state.tool_name, "find_impl")


def build_navigator_graph() -> NavigatorStateGraph:
    """Construct and return the wired StateGraph."""
    g = NavigatorStateGraph()
    g.set_entry_point("router")

    g.add_node("router",        _node_router)
    g.add_node("find_impl",     _node_find_impl)
    g.add_node("trace_lin",     _node_trace_lin)
    g.add_node("blast",         _node_blast)
    g.add_node("explain",       _node_explain)
    g.add_node("semantic_search", _node_semantic)
    g.add_node("format",        _node_format)

    # router → one of the tool nodes (conditional)
    g.add_conditional_edges("router", _route_from_state, {
        "find_impl":      "find_impl",
        "trace_lin":      "trace_lin",
        "blast":          "blast",
        "explain":        "explain",
        "semantic_search":"semantic_search",
    })

    # all tool nodes → format
    for node in ("find_impl", "trace_lin", "blast", "explain", "semantic_search"):
        g.add_edge(node, "format")

    # format → END
    g.add_edge("format", "END")

    return g


# ══════════════════════════════════════════════════════════════════════════════
# Public Navigator interface
# ══════════════════════════════════════════════════════════════════════════════

class Navigator:
    """
    Public interface to the StateGraph.
    Wraps graph.invoke() and exposes direct tool-call methods for CLI use.
    """

    def __init__(self, kg: KnowledgeGraph, result: CartographyResult,
                 api_key: str = ""):
        self.tools  = NavigatorTools(kg, result)
        self.result = result
        self.graph  = build_navigator_graph()
        self._client = None
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if ANTHROPIC_OK and api_key:
            self._client = anthropic.Anthropic(api_key=api_key)

    def interactive_query(self, query: str) -> str:
        """
        Main entry point: run query through the StateGraph, return formatted string.
        Also returns the execution trace in square brackets at the end.
        """
        state  = GraphState(query=query)
        state  = self.graph.invoke(state, self.tools, self._client)
        trace  = " → ".join(state.steps)
        answer = state.final_answer
        method = f"\n  [Route: {state.routing_method} | {trace}]"
        return answer + method

    # ── Direct tool-call methods (for CLI sub-commands) ───────────────────────

    def find_implementation(self, concept: str) -> Dict[str, Any]:
        return self.tools.find_implementation(concept)

    def trace_lineage(self, dataset: str, direction: str = "upstream") -> Dict[str, Any]:
        return self.tools.trace_lineage(dataset, direction)

    def blast_radius(self, module_path: str) -> Dict[str, Any]:
        return self.tools.blast_radius(module_path)

    def explain_module(self, path: str) -> Dict[str, Any]:
        return self.tools.explain_module(path)

    def semantic_search(self, query: str) -> Dict[str, Any]:
        return self.tools.semantic_search(query)
