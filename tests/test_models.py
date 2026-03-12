"""
tests/test_models.py — schema validation tests
Run: python3 -m unittest tests.test_models -v
     pytest tests/test_models.py -v  (when pytest available)
"""
import sys, unittest, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import (
    ValidationError, Language, StorageType, EdgeType,
    ModuleNode, DatasetNode, FunctionNode, TransformationNode,
    GraphEdge, AnalysisTrace, CartographyResult,
)

class TestEnumCoercion(unittest.TestCase):
    def test_language_from_string(self):
        self.assertEqual(Language.coerce("python"), Language.PYTHON)
        self.assertEqual(Language.coerce("SQL"), Language.SQL)
    def test_language_unknown_fallback(self):
        self.assertEqual(Language.coerce("cobol"), Language.UNKNOWN)
        self.assertEqual(Language.coerce(None), Language.UNKNOWN)
    def test_storage_type_coerce(self):
        self.assertEqual(StorageType.coerce("table"), StorageType.TABLE)
        self.assertEqual(StorageType.coerce("nope"), StorageType.UNKNOWN)
    def test_edge_type_valid(self):
        self.assertEqual(EdgeType.coerce("IMPORTS"), EdgeType.IMPORTS)
        self.assertEqual(EdgeType.coerce("produces"), EdgeType.PRODUCES)
    def test_edge_type_invalid_raises(self):
        with self.assertRaises(ValidationError): EdgeType.coerce("INVENTED")

class TestModuleNode(unittest.TestCase):
    def test_basic(self):
        m = ModuleNode(path="src/ingest.py")
        self.assertEqual(m.language, Language.UNKNOWN)
        self.assertEqual(m.imports, [])
    def test_empty_path_raises(self):
        with self.assertRaises(ValidationError): ModuleNode(path="")
    def test_whitespace_path_raises(self):
        with self.assertRaises(ValidationError): ModuleNode(path="   ")
    def test_language_coerced(self):
        m = ModuleNode(path="x.py", language="python")
        self.assertEqual(m.language, Language.PYTHON)
    def test_bad_language_coerced_unknown(self):
        m = ModuleNode(path="x.py", language="brainfuck")
        self.assertEqual(m.language, Language.UNKNOWN)
    def test_negative_complexity_raises(self):
        with self.assertRaises(ValidationError): ModuleNode(path="x.py", complexity_score=-1.0)
    def test_negative_loc_raises(self):
        with self.assertRaises(ValidationError): ModuleNode(path="x.py", loc=-5)
    def test_comment_ratio_clamped_high(self):
        self.assertEqual(ModuleNode(path="x.py", comment_ratio=5.0).comment_ratio, 1.0)
    def test_comment_ratio_clamped_low(self):
        self.assertEqual(ModuleNode(path="x.py", comment_ratio=-1.0).comment_ratio, 0.0)
    def test_none_imports_coerced(self):
        m = ModuleNode(path="x.py", imports=None)
        self.assertEqual(m.imports, [])
    def test_to_dict(self):
        m = ModuleNode(path="src/x.py", language=Language.PYTHON,
                       exports=["fn_a"], loc=50, doc_drift_flag=True)
        d = m.to_dict()
        self.assertEqual(d["language"], "python")
        self.assertIn("fn_a", d["exports"])
        self.assertTrue(d["doc_drift_flag"])
    def test_from_dict_roundtrip(self):
        m = ModuleNode(path="src/x.py", language=Language.SQL, loc=50)
        m2 = ModuleNode.from_dict(m.to_dict())
        self.assertEqual(m2.path, "src/x.py")
        self.assertEqual(m2.language, Language.SQL)
        self.assertEqual(m2.loc, 50)
    def test_js_ts_fields_default_empty(self):
        m = ModuleNode(path="app.js")
        self.assertEqual(m.js_ts_exports, [])

class TestDatasetNode(unittest.TestCase):
    def test_basic(self):
        d = DatasetNode(name="orders")
        self.assertEqual(d.storage_type, StorageType.UNKNOWN)
    def test_empty_name_raises(self):
        with self.assertRaises(ValidationError): DatasetNode(name="")
    def test_storage_type_coerced(self):
        d = DatasetNode(name="events", storage_type="stream")
        self.assertEqual(d.storage_type, StorageType.STREAM)
    def test_none_schema_snapshot_coerced(self):
        d = DatasetNode(name="x", schema_snapshot=None)
        self.assertEqual(d.schema_snapshot, {})
    def test_column_lineage(self):
        d = DatasetNode(name="rev", column_lineage={"total": ["amount", "tax"]})
        self.assertEqual(d.column_lineage["total"], ["amount", "tax"])
    def test_column_lineage_in_to_dict(self):
        d = DatasetNode(name="x", column_lineage={"a": ["b"]})
        self.assertEqual(d.to_dict()["column_lineage"]["a"], ["b"])
    def test_from_dict_roundtrip(self):
        d = DatasetNode(name="customers", storage_type=StorageType.TABLE, is_source_of_truth=True)
        d2 = DatasetNode.from_dict(d.to_dict())
        self.assertTrue(d2.is_source_of_truth)
        self.assertEqual(d2.storage_type, StorageType.TABLE)

class TestFunctionNode(unittest.TestCase):
    def test_basic(self):
        f = FunctionNode(qualified_name="x.py::fn", parent_module="x.py")
        self.assertFalse(f.is_public_api)
    def test_empty_qname_raises(self):
        with self.assertRaises(ValidationError): FunctionNode(qualified_name="", parent_module="x")
    def test_empty_parent_raises(self):
        with self.assertRaises(ValidationError): FunctionNode(qualified_name="x::f", parent_module="")
    def test_negative_complexity_raises(self):
        with self.assertRaises(ValidationError):
            FunctionNode(qualified_name="x::f", parent_module="x.py", complexity=-1)
    def test_to_dict(self):
        f = FunctionNode(qualified_name="x::fn", parent_module="x.py",
                         is_public_api=True, line_number=42, return_type="str")
        d = f.to_dict()
        self.assertTrue(d["is_public_api"])
        self.assertEqual(d["line_number"], 42)
        self.assertEqual(d["return_type"], "str")

class TestTransformationNode(unittest.TestCase):
    def test_basic(self):
        t = TransformationNode(name="t1", source_datasets=["src"], target_datasets=["tgt"])
        self.assertEqual(len(t.source_datasets), 1)
    def test_empty_name_raises(self):
        with self.assertRaises(ValidationError): TransformationNode(name="")
    def test_none_datasets_coerced(self):
        t = TransformationNode(name="t", source_datasets=None, target_datasets=None)
        self.assertEqual(t.source_datasets, [])
    def test_column_mappings(self):
        t = TransformationNode(name="t", column_mappings={"rev": ["amt", "tax"]})
        self.assertEqual(t.column_mappings["rev"], ["amt", "tax"])
    def test_roundtrip(self):
        t = TransformationNode(name="j", source_datasets=["a"], target_datasets=["b"])
        t2 = TransformationNode.from_dict(t.to_dict())
        self.assertEqual(t2.name, "j")
        self.assertEqual(t2.source_datasets, ["a"])

class TestGraphEdge(unittest.TestCase):
    def test_basic(self):
        e = GraphEdge(source="a.py", target="b.py", edge_type=EdgeType.IMPORTS)
        self.assertEqual(e.weight, 1.0)
    def test_string_edge_type_coerced(self):
        e = GraphEdge(source="x", target="y", edge_type="PRODUCES")
        self.assertEqual(e.edge_type, EdgeType.PRODUCES)
    def test_empty_source_raises(self):
        with self.assertRaises(ValidationError):
            GraphEdge(source="", target="y", edge_type=EdgeType.IMPORTS)
    def test_empty_target_raises(self):
        with self.assertRaises(ValidationError):
            GraphEdge(source="x", target="", edge_type=EdgeType.IMPORTS)
    def test_zero_weight_raises(self):
        with self.assertRaises(ValidationError):
            GraphEdge(source="x", target="y", edge_type=EdgeType.IMPORTS, weight=0)
    def test_invalid_type_raises(self):
        with self.assertRaises(ValidationError):
            GraphEdge(source="x", target="y", edge_type="BOGUS")
    def test_to_dict(self):
        e = GraphEdge(source="a", target="b", edge_type=EdgeType.IMPORTS, weight=2.0)
        d = e.to_dict()
        self.assertEqual(d["edge_type"], "IMPORTS")
        self.assertEqual(d["weight"], 2.0)

class TestAnalysisTrace(unittest.TestCase):
    def test_to_jsonl(self):
        t = AnalysisTrace(timestamp="2026-03-09T10:00:00Z", agent="surveyor",
                          action="scan", target="/r", result_summary="ok")
        parsed = json.loads(t.to_jsonl())
        self.assertEqual(parsed["agent"], "surveyor")

class TestCartographyResult(unittest.TestCase):
    def test_defaults(self):
        r = CartographyResult(repo_path="/r", repo_name="r", analysis_timestamp="now")
        self.assertEqual(r.module_nodes, {})
        self.assertEqual(r.errors, [])
    def test_to_dict(self):
        r = CartographyResult(repo_path="/r", repo_name="myrepo", analysis_timestamp="now")
        r.module_nodes["x.py"] = ModuleNode(path="x.py")
        r.pagerank_scores["x.py"] = 0.5
        d = r.to_dict()
        self.assertIn("x.py", d["module_nodes"])
        self.assertAlmostEqual(d["pagerank_scores"]["x.py"], 0.5)

if __name__ == "__main__": unittest.main()
