"""
tests/test_analyzers.py
Run: python3 -m unittest tests.test_analyzers -v
"""
import sys, json, unittest, tempfile, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.analyzers.tree_sitter_analyzer import (
    TreeSitterAnalyzer, PythonASTAnalyzer, JSTSAnalyzer,
    SQLAnalyzerBasic, YAMLAnalyzer, NotebookAnalyzer, LanguageRouter,
)
from src.models import Language

PYTHON_SOURCE = '''\
"""Revenue calculation module."""
import pandas as pd
import numpy as np
from pathlib import Path

def compute_revenue(df, region: str = "global"):
    """Aggregate revenue."""
    if region != "global":
        df = df[df.region == region]
    for month in df.month.unique():
        if df[df.month == month].empty:
            continue
    return df.groupby("customer_id").agg({"amount": "sum"})

class RevenueCalc:
    def __init__(self, config): self.config = config
    def run(self): pass

def _internal(): pass
'''

JS_SOURCE = """\
import React from 'react';
import { api } from '../utils/api';
const lodash = require('lodash');
export function DataTable({ data }) { return null; }
export class DataService { fetchData() { return api.get('/data'); } }
export const formatDate = (d) => d.toISOString();
export default function App() { return null; }
function _internalUtil() {}
"""

SQL_SOURCE = """\
WITH monthly AS (SELECT customer_id, SUM(amount) as total FROM orders o JOIN line_items li ON o.id = li.order_id GROUP BY 1)
CREATE VIEW revenue_summary AS SELECT m.customer_id, c.name, m.total FROM monthly m JOIN customers c ON m.customer_id = c.id;
"""

class TestLanguageRouter(unittest.TestCase):
    def test_python(self): self.assertEqual(LanguageRouter.detect("src/pipeline.py"), Language.PYTHON.value)
    def test_sql(self): self.assertEqual(LanguageRouter.detect("models/orders.sql"), Language.SQL.value)
    def test_yaml(self): self.assertEqual(LanguageRouter.detect("schema.yml"), Language.YAML.value)
    def test_yaml2(self): self.assertEqual(LanguageRouter.detect("schema.yaml"), Language.YAML.value)
    def test_js(self): self.assertEqual(LanguageRouter.detect("app.js"), Language.JAVASCRIPT.value)
    def test_jsx(self): self.assertEqual(LanguageRouter.detect("comp.jsx"), Language.JAVASCRIPT.value)
    def test_ts(self): self.assertEqual(LanguageRouter.detect("service.ts"), Language.TYPESCRIPT.value)
    def test_notebook(self): self.assertEqual(LanguageRouter.detect("nb.ipynb"), Language.NOTEBOOK.value)
    def test_unknown(self): self.assertEqual(LanguageRouter.detect("Makefile"), Language.UNKNOWN.value)
    def test_skip_pycache(self): self.assertTrue(LanguageRouter.should_skip("__pycache__/x.py"))
    def test_skip_git(self): self.assertTrue(LanguageRouter.should_skip(".git/hooks/pre-commit"))
    def test_skip_nodemodules(self): self.assertTrue(LanguageRouter.should_skip("node_modules/lodash/index.js"))
    def test_no_skip_src(self): self.assertFalse(LanguageRouter.should_skip("src/pipeline.py"))

class TestPythonASTAnalyzer(unittest.TestCase):
    def setUp(self):
        self.m, self.fns = PythonASTAnalyzer().analyze("src/revenue_calc.py", PYTHON_SOURCE)
    def test_language(self): self.assertEqual(self.m.language, Language.PYTHON.value)
    def test_docstring(self): self.assertIn("Revenue", self.m.docstring)
    def test_imports(self):
        self.assertIn("pandas", self.m.imports)
        self.assertIn("numpy", self.m.imports)
    def test_public_exports(self):
        self.assertIn("compute_revenue", self.m.exports)
        self.assertIn("RevenueCalc", self.m.exports)
    def test_private_not_exported(self): self.assertNotIn("_internal", self.m.exports)
    def test_functions_extracted(self):
        names = [f.qualified_name.split("::")[-1] for f in self.fns]
        self.assertIn("compute_revenue", names)
    def test_public_api_flagged(self):
        pub = [f for f in self.fns if f.qualified_name.endswith("compute_revenue")]
        self.assertTrue(pub[0].is_public_api)
    def test_private_not_public(self):
        priv = [f for f in self.fns if "_internal" in f.qualified_name]
        self.assertTrue(len(priv) == 1 and not priv[0].is_public_api)
    def test_complexity(self): self.assertGreaterEqual(self.m.complexity_score, 4)
    def test_loc(self): self.assertEqual(self.m.loc, len(PYTHON_SOURCE.splitlines()))
    def test_syntax_error_graceful(self):
        m, fns = PythonASTAnalyzer().analyze("bad.py", "def foo(\n  # unclosed")
        self.assertIn("[PARSE ERROR", m.purpose_statement)
        self.assertEqual(fns, [])

class TestJSTSAnalyzer(unittest.TestCase):
    def setUp(self):
        self.m, self.fns = JSTSAnalyzer().analyze("src/DataTable.jsx", JS_SOURCE)
    def test_language(self): self.assertEqual(self.m.language, Language.JAVASCRIPT.value)
    def test_imports(self):
        self.assertIn("react", self.m.imports)
        self.assertIn("../utils/api", self.m.imports)
        self.assertIn("lodash", self.m.imports)
    def test_named_exports(self):
        self.assertIn("DataTable", self.m.exports)
        self.assertIn("DataService", self.m.exports)
    def test_js_ts_exports_populated(self): self.assertGreaterEqual(len(self.m.js_ts_exports), 2)
    def test_complexity(self): self.assertGreaterEqual(self.m.complexity_score, 1)
    def test_typescript_language(self):
        m, _ = JSTSAnalyzer().analyze("svc.ts", "import { x } from './x';\nexport const y = 1;")
        self.assertEqual(m.language, Language.TYPESCRIPT.value)

class TestSQLAnalyzerBasic(unittest.TestCase):
    def setUp(self): self.m = SQLAnalyzerBasic().analyze("revenue.sql", SQL_SOURCE)
    def test_language(self): self.assertEqual(self.m.language, Language.SQL.value)
    def test_tables_as_imports(self):
        self.assertIn("orders", self.m.imports)
        self.assertIn("line_items", self.m.imports)
        self.assertIn("customers", self.m.imports)
    def test_cte_not_in_imports(self): self.assertNotIn("monthly", self.m.imports)
    def test_view_in_exports(self): self.assertIn("revenue_summary", self.m.exports)

class TestYAMLAnalyzer(unittest.TestCase):
    def test_dbt_models_extracted(self):
        yaml = "version: 2\nmodels:\n  - name: stg_orders\n  - name: stg_customers\n"
        m = YAMLAnalyzer().analyze("schema.yml", yaml)
        self.assertIn("stg_orders", m.exports)
        self.assertIn("stg_customers", m.exports)
    def test_airflow_dag_id(self):
        yaml = "dag_id: daily_pipeline\nschedule: '@daily'\n"
        m = YAMLAnalyzer().analyze("dag.yml", yaml)
        self.assertIn("daily_pipeline", m.exports)

class TestNotebookAnalyzer(unittest.TestCase):
    def _nb(self, cells):
        return json.dumps({"nbformat": 4, "nbformat_minor": 5, "metadata": {},
            "cells": [{"cell_type": "code", "source": c, "metadata": {}, "outputs": []} for c in cells]})
    def test_cells_parsed(self):
        nb = self._nb(["import pandas as pd\n", "def compute_metrics(df):\n    return df.describe()\n"])
        m, fns = NotebookAnalyzer().analyze("nb.ipynb", nb)
        self.assertEqual(m.language, Language.NOTEBOOK.value)
        self.assertIn("pandas", m.imports)
        names = [f.qualified_name.split("::")[-1] for f in fns]
        self.assertIn("compute_metrics", names)
    def test_invalid_json_graceful(self):
        m, _ = NotebookAnalyzer().analyze("bad.ipynb", "NOT JSON {{{")
        self.assertIn("[NOTEBOOK ERROR", m.purpose_statement)

class TestTreeSitterAnalyzerIntegration(unittest.TestCase):
    def test_analyze_python_file(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(PYTHON_SOURCE); f.flush()
            path = f.name
        try:
            m, fns = TreeSitterAnalyzer().analyze_file(path)
            self.assertIsNotNone(m)
            self.assertEqual(m.language, Language.PYTHON.value)
        finally:
            os.unlink(path)
    def test_analyze_js_file(self):
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write(JS_SOURCE); f.flush(); path = f.name
        try:
            m, _ = TreeSitterAnalyzer().analyze_file(path)
            self.assertEqual(m.language, Language.JAVASCRIPT.value)
        finally:
            os.unlink(path)
    def test_unknown_extension_returns_none(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w", delete=False) as f:
            f.write("stuff"); f.flush(); path = f.name
        try:
            m, fns = TreeSitterAnalyzer().analyze_file(path)
            self.assertIsNone(m)
        finally:
            os.unlink(path)
    def test_empty_file_returns_none(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("   \n"); f.flush(); path = f.name
        try:
            m, _ = TreeSitterAnalyzer().analyze_file(path)
            self.assertIsNone(m)
        finally:
            os.unlink(path)
    def test_analyze_repo(self):
        import shutil
        d = tempfile.mkdtemp()
        try:
            Path(d, "ingest.py").write_text("import pandas as pd\ndef load(): pass")
            Path(d, "model.sql").write_text("SELECT * FROM orders")
            Path(d, "schema.yml").write_text("version: 2\nmodels: []")
            modules, _ = TreeSitterAnalyzer().analyze_repo(d)
            langs = [m.language for m in modules]
            self.assertIn(Language.PYTHON.value, langs)
            self.assertIn(Language.SQL.value, langs)
        finally:
            shutil.rmtree(d)

if __name__ == "__main__": unittest.main()
