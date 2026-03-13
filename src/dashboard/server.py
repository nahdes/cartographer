"""
Brownfield Cartographer — Dashboard API Server
Serves the React dashboard and exposes REST + SSE endpoints backed by the
live KnowledgeGraph and Navigator StateGraph.

Endpoints
---------
GET  /                         → dashboard HTML
GET  /api/graph                → module + lineage graph JSON (D3-compatible)
POST /api/risk                 → blast radius, PageRank, change cost (body: {"node_id":"..."})
GET  /api/lineage/<dataset>    → upstream / downstream lineage
GET  /api/modules              → sorted module list with metadata
GET  /api/summary              → repo-level stats card
POST /api/query                → { "q": "..." } → Navigator answer (SSE stream)
"""
from __future__ import annotations
import json
import os
import sys
import time
import threading
from pathlib import Path
from typing import Any, Dict, Generator

# Make sure project root is on path
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

try:
    from flask import Flask, jsonify, request, Response, send_from_directory
    FLASK_OK = True
except ImportError:
    FLASK_OK = False
    raise ImportError("Flask required: pip install flask")

from src.graph.knowledge_graph import KnowledgeGraph
from src.models import CartographyResult

app = Flask(__name__, static_folder=None)

# ── Globals injected by serve() ───────────────────────────────────────────────
_kg:      KnowledgeGraph    = None   # type: ignore[assignment]
_result:  CartographyResult = None   # type: ignore[assignment]
_nav:     Any               = None   # Navigator (lazy-imported to avoid circular)
_api_key: str               = ""
_lock = threading.Lock()


def _get_navigator():
    global _nav
    if _nav is None:
        with _lock:
            if _nav is None:
                from src.agents.navigator import Navigator
                _nav = Navigator(_kg, _result, _api_key)
    return _nav


# ── CORS helper ───────────────────────────────────────────────────────────────
@app.after_request
def _cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


# ══════════════════════════════════════════════════════════════════════════════
# Static dashboard HTML — served from embedded string so no extra files needed
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>Dashboard not found — run build first</h1>", 404


# ══════════════════════════════════════════════════════════════════════════════
# /api/summary  — repo-level stats
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/summary")
def api_summary():
    pr = _result.pagerank_scores or {}
    top_module = max(pr, key=pr.get, default="—") if pr else "—"
    top_module_name = Path(top_module).name if top_module != "—" else "—"

    dead = [p for p, m in _kg.modules.items() if getattr(m, "is_dead_code_candidate", False)]
    drift = [p for p, m in _kg.modules.items() if getattr(m, "doc_drift_flag", False)]
    circ  = _result.circular_dependencies or []

    return jsonify({
        "repo_name":        _result.repo_name,
        "module_count":     len(_kg.modules),
        "dataset_count":    len(_kg.datasets),
        "transform_count":  len(_kg.transformations),
        "function_count":   len(_kg.functions),
        "circular_deps":    len(circ),
        "dead_code":        len(dead),
        "doc_drift":        len(drift),
        "top_module":       top_module_name,
        "top_pagerank":     round(pr.get(top_module, 0), 4) if top_module != "—" else 0,
        "domain_clusters":  list((_result.domain_clusters or {}).keys()),
        "analysis_timestamp": _result.analysis_timestamp,
    })


# ══════════════════════════════════════════════════════════════════════════════
# /api/graph  — D3 force-directed graph payload
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/graph")
def api_graph():
    pr = _result.pagerank_scores or {}
    nodes = []
    edges = []

    # Module nodes
    for path, mod in _kg.modules.items():
        name = Path(path).name
        nodes.append({
            "id":       path,
            "label":    name,
            "type":     "module",
            "language": getattr(mod, "language", "unknown"),
            "pagerank": round(pr.get(path, 0), 5),
            "loc":      getattr(mod, "loc", 0),
            "complexity": round(getattr(mod, "complexity_score", 0), 1),
            "domain":   getattr(mod, "domain_cluster", "other"),
            "dead":     getattr(mod, "is_dead_code_candidate", False),
            "drift":    getattr(mod, "doc_drift_flag", False),
            "purpose":  getattr(mod, "purpose_statement", ""),
            "velocity": getattr(mod, "change_velocity_30d", 0),
        })

    # Dataset nodes
    for name, ds in _kg.datasets.items():
        nodes.append({
            "id":    name,
            "label": name,
            "type":  "dataset",
            "storage": getattr(ds, "storage_label", None) or getattr(ds, "storage_type", "unknown"),
            "owner":   getattr(ds, "owner", ""),
            "source_of_truth": getattr(ds, "is_source_of_truth", False),
            "pagerank": 0,
            "loc": 0,
            "complexity": 0,
            "domain": "data",
            "dead": False,
            "drift": False,
            "purpose": "",
            "velocity": 0,
        })

    # Transformation nodes
    for name, tx in _kg.transformations.items():
        nodes.append({
            "id":    name,
            "label": name,
            "type":  "transform",
            "tx_type": getattr(tx, "transformation_type", ""),
            "source_file": getattr(tx, "source_file", ""),
            "pagerank": 0,
            "loc": 0,
            "complexity": 0,
            "domain": "transform",
            "dead": False,
            "drift": False,
            "purpose": getattr(tx, "transformation_type", ""),
            "velocity": 0,
        })

    # Module import edges
    for u, v, data in _kg.module_graph.edges(data=True):
        edges.append({"source": u, "target": v, "type": "imports",
                      "weight": data.get("weight", 1)})

    # Lineage edges
    for u, v, data in _kg.lineage_graph.edges(data=True):
        edges.append({"source": u, "target": v,
                      "type": data.get("edge_type", "lineage").lower(),
                      "weight": 1})

    return jsonify({"nodes": nodes, "edges": edges})


# ══════════════════════════════════════════════════════════════════════════════
# /api/modules  — sorted list for file nav
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/modules")
def api_modules():
    pr = _result.pagerank_scores or {}
    mods = []
    for path, mod in _kg.modules.items():
        br = _kg.blast_radius(path)
        mods.append({
            "path":      path,
            "name":      Path(path).name,
            "language":  getattr(mod, "language", "?"),
            "loc":       getattr(mod, "loc", 0),
            "pagerank":  round(pr.get(path, 0), 5),
            "blast":     br["total_affected"],
            "domain":    getattr(mod, "domain_cluster", "other"),
            "dead":      getattr(mod, "is_dead_code_candidate", False),
            "drift":     getattr(mod, "doc_drift_flag", False),
            "purpose":   getattr(mod, "purpose_statement", ""),
            "velocity":  getattr(mod, "change_velocity_30d", 0),
        })
    mods.sort(key=lambda m: m["pagerank"], reverse=True)
    return jsonify(mods)


# ══════════════════════════════════════════════════════════════════════════════
# /api/risk  — per-module risk card  (POST body or GET ?node_id=)
# Accepts path-param style via query string to avoid Flask slash-routing issues
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/risk", methods=["GET", "POST", "OPTIONS"])
def api_risk():
    if request.method == "OPTIONS":
        return Response(status=200)
    if request.method == "POST":
        body = request.get_json(silent=True) or {}
        node_id = body.get("node_id") or body.get("id") or ""
    else:
        node_id = request.args.get("node_id") or request.args.get("id") or ""
    if not node_id:
        return jsonify({"error": "Missing node_id"}), 400
    pr = _result.pagerank_scores or {}
    br = _kg.blast_radius(node_id)
    mod = _kg.modules.get(node_id)
    ds  = _kg.datasets.get(node_id)
    tx  = _kg.transformations.get(node_id)

    # For dataset/transform nodes count lineage descendants as blast proxy
    if mod:
        dependents = br["downstream_modules"]
        blast_score = min(100, int(len(dependents) * 12 + pr.get(node_id, 0) * 300))
        loc         = getattr(mod, "loc", 0)
        complexity  = round(getattr(mod, "complexity_score", 0), 1)
        velocity    = getattr(mod, "change_velocity_30d", 0)
        dead        = getattr(mod, "is_dead_code_candidate", False)
        drift       = getattr(mod, "doc_drift_flag", False)
        purpose     = getattr(mod, "purpose_statement", "")
        domain      = getattr(mod, "domain_cluster", "other")
        node_type   = "module"
    elif ds or tx:
        obj = ds or tx
        import networkx as nx
        try:
            desc = list(nx.descendants(_kg.lineage_graph, node_id))
        except Exception:
            desc = []
        dependents  = desc
        blast_score = min(100, len(desc) * 20)
        loc, complexity, velocity = 0, 0.0, 0
        dead, drift = False, False
        purpose     = getattr(obj, "transformation_type", "") if tx else getattr(obj, "storage_type", "")
        domain      = "data"
        node_type   = "dataset" if ds else "transform"
    else:
        dependents, blast_score = [], 0
        loc, complexity, velocity = 0, 0.0, 0
        dead, drift, purpose, domain, node_type = False, False, "", "unknown", "unknown"

    risk_level = "low"
    if blast_score > 60 or len(dependents) >= 5:
        risk_level = "high"
    elif blast_score > 30 or len(dependents) >= 2:
        risk_level = "medium"

    label = Path(node_id).name if "/" in node_id or "\\" in node_id else node_id
    return jsonify({
        "node_id":     node_id,
        "name":        label,
        "type":        node_type,
        "blast_score": blast_score,
        "blast_raw":   len(dependents),
        "pagerank":    round(pr.get(node_id, 0), 5),
        "risk_level":  risk_level,
        "dependents":  dependents[:10],
        "dep_datasets": br.get("downstream_datasets", [])[:5],
        "loc":         loc,
        "complexity":  complexity,
        "velocity":    velocity,
        "dead":        dead,
        "drift":       drift,
        "purpose":     purpose,
        "domain":      domain,
        "change_cost": (
            "Critical — final output" if risk_level == "high" else
            f"Medium — {len(dependents)} dependents" if risk_level == "medium" else
            f"Low — {len(dependents)} dependents"
        ),
    })


# ══════════════════════════════════════════════════════════════════════════════
# /api/lineage/<dataset>  — lineage chain
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/lineage/<dataset>")
def api_lineage(dataset):
    direction = request.args.get("direction", "upstream")
    result = _kg.trace_lineage(dataset, direction)

    # Enrich nodes with column mappings
    enriched = []
    for name in result.get("nodes", []):
        info: Dict[str, Any] = {"name": name}
        if name in _kg.transformations:
            tx = _kg.transformations[name]
            info["type"]            = "transform"
            info["column_mappings"] = getattr(tx, "column_mappings", {})
            info["source_file"]     = getattr(tx, "source_file", "")
        elif name in _kg.datasets:
            ds = _kg.datasets[name]
            info["type"]           = "dataset"
            info["storage_type"]   = getattr(ds, "storage_label", None) or getattr(ds, "storage_type", "unknown")
            info["column_lineage"] = getattr(ds, "column_lineage", {})
        enriched.append(info)

    result["enriched_nodes"] = enriched
    return jsonify(result)


# ══════════════════════════════════════════════════════════════════════════════
# /api/query  — POST, SSE-streamed Navigator answer
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/query", methods=["POST", "OPTIONS"])
def api_query():
    if request.method == "OPTIONS":
        return Response(status=200)

    body = request.get_json(silent=True) or {}
    query = (body.get("q") or body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Missing query field 'q'"}), 400

    def generate() -> Generator[str, None, None]:
        nav = _get_navigator()
        try:
            # Emit thinking indicator
            yield f"data: {json.dumps({'type': 'thinking', 'text': '...'})}\n\n"
            answer = nav.interactive_query(query)

            # Split into chunks for a streaming feel
            words = answer.split()
            chunk_size = 6
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                if i + chunk_size < len(words):
                    chunk += " "
                yield f"data: {json.dumps({'type': 'token', 'text': chunk})}\n\n"
                time.sleep(0.04)

            yield f"data: {json.dumps({'type': 'done', 'full': answer})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'text': str(exc)})}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ══════════════════════════════════════════════════════════════════════════════
# Launch helper
# ══════════════════════════════════════════════════════════════════════════════

def serve(kg: KnowledgeGraph, result: CartographyResult,
          api_key: str = "", host: str = "127.0.0.1", port: int = 7842,
          open_browser: bool = True):
    """Inject live KG + result into Flask globals and start the server."""
    global _kg, _result, _api_key
    _kg      = kg
    _result  = result
    _api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    url = f"http://{host}:{port}"
    print(f"\n🌐  Dashboard: {url}")
    print(f"   Press Ctrl+C to stop\n")

    if open_browser:
        import threading, webbrowser
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)