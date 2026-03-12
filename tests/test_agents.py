"""
tests/test_agents.py — Surveyor + Hydrologist integration tests
Run: python3 -m unittest tests.test_agents -v
"""
import sys, unittest, textwrap, tempfile, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.graph.knowledge_graph import KnowledgeGraph
from src.agents.surveyor import Surveyor
from src.agents.hydrologist import Hydrologist
from src.models import Language

def make_repo(root):
    src = root / "src"; src.mkdir()
    models = root / "models"; models.mkdir()
    (src / "ingest.py").write_text(textwrap.dedent("""\
        \"\"\"Kafka consumer for raw event ingestion.\"\"\"
        import pandas as pd
        def load_events(path: str):
            return pd.read_csv(path)
        def save_enriched(df, out: str):
            df.to_parquet(out)
    """))
    (src / "transform.py").write_text(textwrap.dedent("""\
        \"\"\"Revenue transforms.\"\"\"
        import pandas as pd
        from src.ingest import load_events
        def compute_daily_metrics(events_path: str, out: str):
            df = pd.read_parquet(events_path)
            df.groupby('date').agg({'revenue': 'sum'}).to_csv(out)
    """))
    (models / "orders.sql").write_text(textwrap.dedent("""\
        SELECT o.id, o.amount, c.name
        FROM stg_orders o JOIN stg_customers c ON o.customer_id = c.id
        WHERE o.status = 'completed'
    """))
    (root / "schema.yml").write_text("version: 2\nmodels:\n  - name: stg_orders\n  - name: stg_customers\n")
    return root

class TestSurveyor(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        make_repo(self.tmpdir)
        self.kg = KnowledgeGraph()
        self.surveyor = Surveyor(self.kg)
        self.result = self.surveyor.run(str(self.tmpdir))
    def tearDown(self): shutil.rmtree(self.tmpdir, ignore_errors=True)
    def test_modules_found(self):
        self.assertGreaterEqual(self.result["module_count"], 3)
    def test_python_modules_registered(self):
        langs = [m.language for m in self.kg.modules.values()]
        self.assertIn(Language.PYTHON.value, langs)
    def test_functions_extracted(self):
        self.assertGreaterEqual(self.result["function_count"], 2)
    def test_import_graph_built(self):
        self.assertGreaterEqual(self.kg.module_graph.number_of_nodes(), 2)
    def test_pagerank_computed(self):
        pr = self.result["pagerank"]
        self.assertIsInstance(pr, dict)
        self.assertGreaterEqual(len(pr), 1)
        self.assertTrue(all(0 <= v <= 1 for v in pr.values()))
    def test_no_circular_deps(self):
        self.assertEqual(self.result["circular_deps"], [])
    def test_traces_populated(self):
        self.assertGreaterEqual(len(self.surveyor.traces), 1)
        self.assertEqual(self.surveyor.traces[0].agent, "surveyor")

class TestHydrologist(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        make_repo(self.tmpdir)
        self.kg = KnowledgeGraph()
        Surveyor(self.kg).run(str(self.tmpdir))
        self.hydro = Hydrologist(self.kg)
        self.result = self.hydro.run(str(self.tmpdir))
    def tearDown(self): shutil.rmtree(self.tmpdir, ignore_errors=True)
    def test_datasets_detected(self): self.assertGreaterEqual(len(self.kg.datasets), 1)
    def test_python_reads_detected(self):
        names = list(self.kg.datasets.keys())
        self.assertTrue(any("csv" in d or "parquet" in d or "events" in d for d in names))
    def test_sql_tables_detected(self):
        names = [d.lower() for d in self.kg.datasets]
        self.assertTrue(any("stg_orders" in d or "orders" in d for d in names))
    def test_sources_sinks_are_lists(self):
        self.assertIsInstance(self.result.get("sources",[]), list)
        self.assertIsInstance(self.result.get("sinks",[]), list)
    def test_transformations_registered(self): self.assertGreaterEqual(len(self.kg.transformations), 1)
    def test_traces_populated(self):
        self.assertGreaterEqual(len(self.hydro.traces), 1)
        self.assertEqual(self.hydro.traces[0].agent, "hydrologist")
    def test_lineage_graph_has_edges(self):
        self.assertGreaterEqual(self.kg.lineage_graph.number_of_edges(), 1)

class TestCircularDepDetection(unittest.TestCase):
    def test_cycle_reported(self):
        d = Path(tempfile.mkdtemp())
        try:
            (d/"module_a.py").write_text("from module_b import helper\ndef run(): pass\n")
            (d/"module_b.py").write_text("from module_a import run\ndef helper(): pass\n")
            kg = KnowledgeGraph(); s = Surveyor(kg); result = s.run(str(d))
            self.assertIsInstance(result["circular_deps"], list)
        finally:
            shutil.rmtree(d, ignore_errors=True)

if __name__ == "__main__": unittest.main()
