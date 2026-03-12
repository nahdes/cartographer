"""
tests/test_pydantic_schemas.py — Pydantic v2 schemas + fallback shims (Gap 5)
Run: python3 -m unittest tests.test_pydantic_schemas -v
"""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.pydantic_schemas import (
    PYDANTIC_AVAILABLE, ModuleNodeSchema, DatasetNodeSchema,
    FunctionNodeSchema, TransformationNodeSchema, GraphEdgeSchema,
)
from src.models import (
    ModuleNode, DatasetNode, FunctionNode, TransformationNode, GraphEdge,
    EdgeType, Language, ValidationError,
)


# ── Always-valid shim tests (work regardless of pydantic install) ─────────────

class TestSchemaAvailability(unittest.TestCase):
    def test_pydantic_available_is_bool(self):
        self.assertIsInstance(PYDANTIC_AVAILABLE, bool)

    def test_all_schema_classes_importable(self):
        for cls in (ModuleNodeSchema, DatasetNodeSchema, FunctionNodeSchema,
                    TransformationNodeSchema, GraphEdgeSchema):
            self.assertIsNotNone(cls)


@unittest.skipUnless(PYDANTIC_AVAILABLE, "pydantic not installed")
class TestModuleNodeSchema(unittest.TestCase):
    """Full Pydantic v2 validation for ModuleNodeSchema."""

    def test_valid_minimal(self):
        s = ModuleNodeSchema(path="src/x.py")
        self.assertEqual(s.path, "src/x.py")
        self.assertEqual(s.language, "unknown")

    def test_path_whitespace_stripped(self):
        s = ModuleNodeSchema(path="  src/x.py  ")
        self.assertEqual(s.path, "src/x.py")

    def test_empty_path_raises(self):
        from pydantic import ValidationError as PVE
        with self.assertRaises(PVE):
            ModuleNodeSchema(path="")

    def test_blank_path_raises(self):
        from pydantic import ValidationError as PVE
        with self.assertRaises(PVE):
            ModuleNodeSchema(path="   ")

    def test_comment_ratio_clamp_high(self):
        s = ModuleNodeSchema(path="x.py", comment_ratio=5.0)
        self.assertEqual(s.comment_ratio, 1.0)

    def test_comment_ratio_clamp_low(self):
        s = ModuleNodeSchema(path="x.py", comment_ratio=-0.5)
        self.assertEqual(s.comment_ratio, 0.0)

    def test_complexity_score_ge_zero(self):
        from pydantic import ValidationError as PVE
        with self.assertRaises(PVE):
            ModuleNodeSchema(path="x.py", complexity_score=-1.0)

    def test_loc_ge_zero(self):
        from pydantic import ValidationError as PVE
        with self.assertRaises(PVE):
            ModuleNodeSchema(path="x.py", loc=-10)

    def test_imports_none_becomes_list(self):
        s = ModuleNodeSchema(path="x.py", imports=None)
        self.assertEqual(s.imports, [])

    def test_exports_list_preserved(self):
        s = ModuleNodeSchema(path="x.py", exports=["foo", "bar"])
        self.assertEqual(s.exports, ["foo", "bar"])

    def test_unknown_field_ignored(self):
        # extra="ignore" — should not raise
        s = ModuleNodeSchema(path="x.py", nonexistent_field="value")
        self.assertFalse(hasattr(s, "nonexistent_field"))

    def test_model_dump_returns_dict(self):
        s = ModuleNodeSchema(path="x.py", loc=50)
        d = s.model_dump()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["path"], "x.py")
        self.assertEqual(d["loc"], 50)

    def test_model_validate_from_dict(self):
        d = {"path": "src/y.py", "loc": 100, "language": "python"}
        s = ModuleNodeSchema.model_validate(d)
        self.assertEqual(s.path, "src/y.py")
        self.assertEqual(s.loc, 100)

    def test_model_json_schema_returned(self):
        schema = ModuleNodeSchema.model_json_schema()
        self.assertIsInstance(schema, dict)
        self.assertIn("properties", schema)

    def test_validate_assignment_enforced(self):
        from pydantic import ValidationError as PVE
        s = ModuleNodeSchema(path="x.py")
        with self.assertRaises(PVE):
            s.loc = -1  # ge=0 violated


@unittest.skipUnless(PYDANTIC_AVAILABLE, "pydantic not installed")
class TestDatasetNodeSchema(unittest.TestCase):
    def test_valid_minimal(self):
        s = DatasetNodeSchema(name="orders")
        self.assertEqual(s.name, "orders")

    def test_empty_name_raises(self):
        from pydantic import ValidationError as PVE
        with self.assertRaises(PVE):
            DatasetNodeSchema(name="")

    def test_schema_snapshot_none_becomes_dict(self):
        s = DatasetNodeSchema(name="t", schema_snapshot=None)
        self.assertEqual(s.schema_snapshot, {})

    def test_column_lineage_preserved(self):
        cl = {"total": ["amount", "tax"]}
        s = DatasetNodeSchema(name="t", column_lineage=cl)
        self.assertEqual(s.column_lineage, cl)

    def test_model_validate_roundtrip(self):
        orig = DatasetNode(name="orders", source_file="q.sql")
        s = DatasetNodeSchema.model_validate(orig.to_dict())
        self.assertEqual(s.name, "orders")
        self.assertEqual(s.source_file, "q.sql")


@unittest.skipUnless(PYDANTIC_AVAILABLE, "pydantic not installed")
class TestFunctionNodeSchema(unittest.TestCase):
    def test_valid(self):
        s = FunctionNodeSchema(qualified_name="x.py::fn", parent_module="x.py")
        self.assertEqual(s.qualified_name, "x.py::fn")

    def test_empty_qualified_name_raises(self):
        from pydantic import ValidationError as PVE
        with self.assertRaises(PVE):
            FunctionNodeSchema(qualified_name="", parent_module="x.py")

    def test_empty_parent_module_raises(self):
        from pydantic import ValidationError as PVE
        with self.assertRaises(PVE):
            FunctionNodeSchema(qualified_name="x.py::fn", parent_module="")

    def test_line_number_ge_zero(self):
        from pydantic import ValidationError as PVE
        with self.assertRaises(PVE):
            FunctionNodeSchema(qualified_name="x.py::f", parent_module="x.py",
                               line_number=-5)


@unittest.skipUnless(PYDANTIC_AVAILABLE, "pydantic not installed")
class TestTransformationNodeSchema(unittest.TestCase):
    def test_valid_minimal(self):
        s = TransformationNodeSchema(name="calc_revenue")
        self.assertEqual(s.name, "calc_revenue")

    def test_empty_name_raises(self):
        from pydantic import ValidationError as PVE
        with self.assertRaises(PVE):
            TransformationNodeSchema(name="")

    def test_source_datasets_preserved(self):
        s = TransformationNodeSchema(name="t", source_datasets=["orders", "customers"])
        self.assertEqual(s.source_datasets, ["orders", "customers"])

    def test_column_mappings_preserved(self):
        cm = {"total": ["amount"]}
        s = TransformationNodeSchema(name="t", column_mappings=cm)
        self.assertEqual(s.column_mappings, cm)

    def test_none_lists_coerced(self):
        s = TransformationNodeSchema(name="t", source_datasets=None,
                                     target_datasets=None, line_range=None)
        self.assertEqual(s.source_datasets, [])
        self.assertEqual(s.target_datasets, [])
        self.assertEqual(s.line_range, [])

    def test_model_validate_from_transformation_node(self):
        t = TransformationNode(name="tx", source_datasets=["a"],
                               target_datasets=["b"],
                               column_mappings={"x": ["y"]})
        s = TransformationNodeSchema.model_validate(t.to_dict())
        self.assertEqual(s.name, "tx")
        self.assertEqual(s.column_mappings, {"x": ["y"]})


@unittest.skipUnless(PYDANTIC_AVAILABLE, "pydantic not installed")
class TestGraphEdgeSchema(unittest.TestCase):
    def test_valid(self):
        s = GraphEdgeSchema(source="a", target="b", edge_type="IMPORTS")
        self.assertEqual(s.source, "a")
        self.assertEqual(s.weight, 1.0)

    def test_zero_weight_raises(self):
        from pydantic import ValidationError as PVE
        with self.assertRaises(PVE):
            GraphEdgeSchema(source="a", target="b", edge_type="IMPORTS", weight=0)

    def test_negative_weight_raises(self):
        from pydantic import ValidationError as PVE
        with self.assertRaises(PVE):
            GraphEdgeSchema(source="a", target="b", edge_type="X", weight=-1.0)

    def test_empty_source_raises(self):
        from pydantic import ValidationError as PVE
        with self.assertRaises(PVE):
            GraphEdgeSchema(source="", target="b", edge_type="X")


@unittest.skipUnless(PYDANTIC_AVAILABLE, "pydantic not installed")
class TestSchemaRoundtrips(unittest.TestCase):
    """Full dataclass → schema → dump → re-validate roundtrips."""

    def test_module_node_roundtrip(self):
        node = ModuleNode(path="src/z.py", language=Language.PYTHON.value,
                          loc=200, exports=["foo"], imports=["os", "sys"])
        s = ModuleNodeSchema.model_validate(node.to_dict())
        d = s.model_dump()
        s2 = ModuleNodeSchema.model_validate(d)
        self.assertEqual(s2.path, "src/z.py")
        self.assertEqual(s2.exports, ["foo"])

    def test_dataset_node_roundtrip(self):
        node = DatasetNode(name="events",
                           column_lineage={"user_id": ["raw_user_id"]})
        s  = DatasetNodeSchema.model_validate(node.to_dict())
        d  = s.model_dump()
        s2 = DatasetNodeSchema.model_validate(d)
        self.assertEqual(s2.column_lineage, {"user_id": ["raw_user_id"]})

    def test_transformation_node_roundtrip(self):
        node = TransformationNode(name="enrich",
                                  source_datasets=["raw"],
                                  target_datasets=["clean"],
                                  column_mappings={"id": ["raw_id"]})
        s  = TransformationNodeSchema.model_validate(node.to_dict())
        d  = s.model_dump()
        s2 = TransformationNodeSchema.model_validate(d)
        self.assertEqual(s2.column_mappings, {"id": ["raw_id"]})


# ── Shim tests — always run, verify fallback works without pydantic ───────────

class TestDataclassIntegration(unittest.TestCase):
    """
    These tests verify that the dataclass models themselves satisfy the same
    constraints that the Pydantic schemas enforce — ensuring consistent
    behaviour whether or not pydantic is installed.
    """

    def test_module_node_comment_ratio_clamp(self):
        m = ModuleNode(path="x.py", comment_ratio=5.0)
        self.assertEqual(m.comment_ratio, 1.0)

    def test_module_node_empty_path_raises(self):
        with self.assertRaises(ValidationError):
            ModuleNode(path="")

    def test_graph_edge_zero_weight_raises(self):
        with self.assertRaises(ValidationError):
            GraphEdge(source="a", target="b", edge_type=EdgeType.IMPORTS, weight=0)

    def test_dataset_node_empty_name_raises(self):
        with self.assertRaises(ValidationError):
            DatasetNode(name="")

    def test_function_node_empty_qualified_name_raises(self):
        with self.assertRaises(ValidationError):
            FunctionNode(qualified_name="", parent_module="x.py")

    def test_transformation_node_empty_name_raises(self):
        with self.assertRaises(ValidationError):
            TransformationNode(name="")

    def test_module_node_to_dict_keys(self):
        m = ModuleNode(path="x.py")
        d = m.to_dict()
        for key in ("path", "language", "loc", "exports", "imports",
                    "complexity_score", "comment_ratio", "pagerank_score"):
            self.assertIn(key, d)

    def test_dataset_node_to_dict_has_column_lineage(self):
        d = DatasetNode(name="t", column_lineage={"a": ["b"]}).to_dict()
        self.assertIn("column_lineage", d)
        self.assertEqual(d["column_lineage"], {"a": ["b"]})

    def test_transformation_node_to_dict_has_column_mappings(self):
        d = TransformationNode(name="t", column_mappings={"x": ["y"]}).to_dict()
        self.assertIn("column_mappings", d)


if __name__ == "__main__":
    unittest.main()
