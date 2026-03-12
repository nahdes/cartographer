"""
tests/test_navigator_graph.py — StateGraph, typed state, all 5 tools (Gaps 3 & 4)
Run: python3 -m unittest tests.test_navigator_graph -v
"""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.agents.navigator import (
    GraphState, NavigatorStateGraph, NavigatorTools, Navigator,
    build_navigator_graph, _node_router, _node_find_impl, _node_format,
    _route_from_state, TOOL_SCHEMAS,
    _fmt_find, _fmt_lineage, _fmt_blast, _fmt_explain, _fmt_semantic,
)
from src.graph.knowledge_graph import KnowledgeGraph
from src.models import (
    CartographyResult, ModuleNode, DatasetNode, TransformationNode, Language,
)
from datetime import datetime, timezone


def _make_result(modules=None, datasets=None):
    r = CartographyResult(
        repo_path="/tmp/test", repo_name="test",
        analysis_timestamp=datetime.now(timezone.utc).isoformat(),
    )
    if modules:
        for m in modules:
            r.module_nodes[m.path] = m
            r.pagerank_scores[m.path] = 0.1
    return r


def _make_kg(modules=None, datasets=None, transformations=None):
    kg = KnowledgeGraph()
    for m in (modules or []):
        kg.add_module(m)
    for d in (datasets or []):
        kg.add_dataset(d)
    for t in (transformations or []):
        kg.add_transformation(t)
    return kg


class TestGraphState(unittest.TestCase):
    def test_default_state(self):
        s = GraphState()
        self.assertEqual(s.query, "")
        self.assertEqual(s.steps, [])
        self.assertEqual(s.tool_params, {})

    def test_add_step(self):
        s = GraphState(query="test")
        s.add_step("router→find_impl (keyword)")
        self.assertEqual(len(s.steps), 1)
        self.assertIn("router", s.steps[0])

    def test_state_mutation(self):
        s = GraphState(query="where is revenue?")
        s.tool_name = "find_implementation"
        s.routing_method = "keyword"
        self.assertEqual(s.tool_name, "find_implementation")


class TestToolSchemas(unittest.TestCase):
    def test_all_five_tools_present(self):
        names = [t["name"] for t in TOOL_SCHEMAS]
        self.assertIn("find_implementation", names)
        self.assertIn("trace_lineage", names)
        self.assertIn("blast_radius", names)
        self.assertIn("explain_module", names)
        self.assertIn("semantic_search", names)

    def test_schemas_have_required_fields(self):
        for schema in TOOL_SCHEMAS:
            self.assertIn("name", schema)
            self.assertIn("description", schema)
            self.assertIn("input_schema", schema)
            self.assertIn("properties", schema["input_schema"])


class TestConditionalRouting(unittest.TestCase):
    def test_route_find_impl(self):
        s = GraphState(tool_name="find_implementation")
        self.assertEqual(_route_from_state(s), "find_impl")

    def test_route_trace_lineage(self):
        s = GraphState(tool_name="trace_lineage")
        self.assertEqual(_route_from_state(s), "trace_lin")

    def test_route_blast_radius(self):
        s = GraphState(tool_name="blast_radius")
        self.assertEqual(_route_from_state(s), "blast")

    def test_route_explain_module(self):
        s = GraphState(tool_name="explain_module")
        self.assertEqual(_route_from_state(s), "explain")

    def test_route_semantic_search(self):
        s = GraphState(tool_name="semantic_search")
        self.assertEqual(_route_from_state(s), "semantic_search")

    def test_route_unknown_defaults_to_find(self):
        s = GraphState(tool_name="unknown_tool")
        self.assertEqual(_route_from_state(s), "find_impl")


class TestKeywordRouter(unittest.TestCase):
    def _route(self, query):
        s = GraphState(query=query)
        result = _node_router(s, None, None)  # no tools or api needed for keyword
        return result.tool_name, result.routing_method

    def test_find_implementation_keyword(self):
        tool, method = self._route("where is the revenue calculation?")
        self.assertEqual(tool, "find_implementation")
        self.assertEqual(method, "keyword")

    def test_trace_lineage_keyword(self):
        tool, method = self._route("trace the lineage of orders dataset")
        self.assertEqual(tool, "trace_lineage")
        self.assertEqual(method, "keyword")

    def test_blast_radius_keyword(self):
        tool, method = self._route("what is the blast radius of utils.py?")
        self.assertEqual(tool, "blast_radius")
        self.assertEqual(method, "keyword")

    def test_explain_keyword(self):
        tool, method = self._route("explain the purpose of transform.py")
        self.assertEqual(tool, "explain_module")
        self.assertEqual(method, "keyword")

    def test_semantic_search_keyword(self):
        tool, method = self._route("what is similar to the revenue module?")
        self.assertEqual(tool, "semantic_search")
        self.assertEqual(method, "keyword")

    def test_fallback_for_unknown(self):
        tool, method = self._route("xyzzy plugh")
        self.assertEqual(method, "fallback")


class TestStateGraphWiring(unittest.TestCase):
    def _make_minimal(self):
        m = ModuleNode(path="src/revenue.py", language=Language.PYTHON.value,
                       purpose_statement="Computes revenue metrics.",
                       exports=["compute_revenue"])
        kg = _make_kg(modules=[m])
        result = _make_result(modules=[m])
        return kg, result

    def test_graph_has_all_nodes(self):
        g = build_navigator_graph()
        for node in ("router", "find_impl", "trace_lin", "blast",
                     "explain", "semantic_search", "format"):
            self.assertIn(node, g._nodes)

    def test_graph_has_conditional_edge_from_router(self):
        g = build_navigator_graph()
        self.assertIn("router", g._cond)

    def test_graph_invoke_find_implementation(self):
        kg, result = self._make_minimal()
        tools = NavigatorTools(kg, result)
        g     = build_navigator_graph()
        state = GraphState(query="where is revenue?",
                           tool_name="find_implementation",
                           tool_params={"concept": "revenue"})
        # Skip router — inject state directly at find_impl
        state = g._nodes["find_impl"](state, tools, None)
        state = g._nodes["format"](state, tools, None)
        self.assertNotEqual(state.final_answer, "")
        self.assertIn("revenue", state.final_answer.lower())

    def test_graph_invoke_explain(self):
        kg, result = self._make_minimal()
        tools = NavigatorTools(kg, result)
        g     = build_navigator_graph()
        state = GraphState(query="explain src/revenue.py",
                           tool_name="explain_module",
                           tool_params={"path": "src/revenue.py"})
        state = g._nodes["explain"](state, tools, None)
        state = g._nodes["format"](state, tools, None)
        self.assertIn("revenue.py", state.final_answer)

    def test_graph_invoke_blast(self):
        kg, result = self._make_minimal()
        tools = NavigatorTools(kg, result)
        g     = build_navigator_graph()
        state = GraphState(query="blast radius of src/revenue.py",
                           tool_name="blast_radius",
                           tool_params={"module_path": "src/revenue.py"})
        state = g._nodes["blast"](state, tools, None)
        state = g._nodes["format"](state, tools, None)
        self.assertIn("blast radius", state.final_answer.lower())

    def test_full_interactive_query(self):
        kg, result = self._make_minimal()
        nav    = Navigator(kg, result)  # no API key — keyword routing
        answer = nav.interactive_query("where is revenue implemented?")
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 20)
        # answer should contain routing info
        self.assertIn("Route:", answer)

    def test_trace_lineage_in_graph(self):
        kg = _make_kg(
            datasets=[DatasetNode(name="raw_events"), DatasetNode(name="enriched_events")],
            transformations=[TransformationNode(name="enrich",
                source_datasets=["raw_events"], target_datasets=["enriched_events"])],
        )
        result = _make_result()
        tools = NavigatorTools(kg, result)
        g = build_navigator_graph()
        state = GraphState(tool_name="trace_lineage",
                           tool_params={"dataset": "enriched_events", "direction": "upstream"})
        state = g._nodes["trace_lin"](state, tools, None)
        state = g._nodes["format"](state, tools, None)
        self.assertIn("enriched_events", state.final_answer)


class TestNavigatorTools(unittest.TestCase):
    def setUp(self):
        modules = [
            ModuleNode(path="src/ingest.py",   language=Language.PYTHON.value,
                       purpose_statement="Kafka consumer for raw events.",
                       exports=["load_events"], domain_cluster="ingestion"),
            ModuleNode(path="src/transform.py", language=Language.PYTHON.value,
                       purpose_statement="Revenue computation and aggregation.",
                       exports=["compute_revenue"], domain_cluster="transformation"),
        ]
        self.kg     = _make_kg(modules=modules)
        self.result = _make_result(modules=modules)
        self.tools  = NavigatorTools(self.kg, self.result)

    def test_find_implementation_returns_match(self):
        r = self.tools.find_implementation("revenue")
        self.assertIn("matches", r)
        paths = [m["path"] for m in r["matches"]]
        self.assertIn("src/transform.py", paths)

    def test_find_implementation_has_analysis_method(self):
        r = self.tools.find_implementation("ingestion")
        self.assertIn("analysis_method", r)

    def test_explain_module_found(self):
        r = self.tools.explain_module("src/ingest.py")
        self.assertNotIn("error", r)
        self.assertEqual(r["path"], "src/ingest.py")
        self.assertIn("purpose", r)

    def test_explain_module_not_found(self):
        r = self.tools.explain_module("nonexistent.py")
        self.assertIn("error", r)

    def test_blast_radius_returns_dict(self):
        r = self.tools.blast_radius("src/ingest.py")
        self.assertIn("total_affected", r)
        self.assertIn("downstream_modules", r)

    def test_semantic_search_returns_result(self):
        r = self.tools.semantic_search("revenue computation")
        self.assertIn("matches", r)
        self.assertIn("analysis_method", r)

    def test_direct_navigator_methods(self):
        nav = Navigator(self.kg, self.result)
        r1 = nav.find_implementation("revenue")
        r2 = nav.blast_radius("src/ingest.py")
        r3 = nav.semantic_search("ingestion")
        self.assertIn("matches", r1)
        self.assertIn("total_affected", r2)
        self.assertIn("matches", r3)


class TestFormatters(unittest.TestCase):
    def test_fmt_find_with_results(self):
        r = {"concept": "revenue", "total_matches": 2,
             "analysis_method": "tfidf",
             "matches": [{"path": "src/x.py", "score": 0.9,
                          "purpose": "Compute revenue", "reasons": []}]}
        out = _fmt_find(r)
        self.assertIn("revenue", out)
        self.assertIn("src/x.py", out)

    def test_fmt_find_no_results(self):
        r = {"concept": "xyz", "total_matches": 0, "analysis_method": "x", "matches": []}
        self.assertIn("No matches", _fmt_find(r))

    def test_fmt_lineage_not_found(self):
        r = {"dataset": "orders", "found": False, "direction": "upstream"}
        self.assertIn("not found", _fmt_lineage(r))

    def test_fmt_lineage_found(self):
        r = {"dataset": "orders", "found": True, "direction": "upstream",
             "nodes": ["raw_events"], "edges": [], "enriched_nodes": [],
             "analysis_method": "graph"}
        out = _fmt_lineage(r)
        self.assertIn("orders", out)
        self.assertIn("raw_events", out)

    def test_fmt_blast(self):
        r = {"source": "utils.py", "total_affected": 3,
             "formatted_downstream": [{"path": "pipeline.py", "depth": 1}],
             "analysis_method": "BFS"}
        out = _fmt_blast(r)
        self.assertIn("utils.py", out)
        self.assertIn("3", out)

    def test_fmt_explain(self):
        r = {"path": "src/x.py", "purpose": "Test module", "domain": "utils",
             "language": "python", "loc": 100, "complexity_score": 5.0,
             "pagerank": 0.01, "blast_radius_count": 2, "doc_drift": False,
             "analysis_method": "AST"}
        out = _fmt_explain(r)
        self.assertIn("src/x.py", out)
        self.assertIn("Test module", out)

    def test_fmt_semantic_no_results(self):
        r = {"query": "x", "matches": [], "analysis_method": "tfidf"}
        self.assertIn("No semantically", _fmt_semantic(r))

    def test_fmt_semantic_with_results(self):
        r = {"query": "revenue", "matches": [{"path": "src/x.py", "score": 0.8,
             "purpose": "Revenue calc", "domain": "transform"}],
             "total": 1, "analysis_method": "tfidf"}
        out = _fmt_semantic(r)
        self.assertIn("src/x.py", out)


if __name__ == "__main__":
    unittest.main()
