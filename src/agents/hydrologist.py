"""
Agent 2: The Hydrologist — Data Flow & Lineage Analyst
Constructs the data lineage DAG across Python, SQL, YAML, and notebooks.
Column-level lineage is extracted from SQL SELECT clauses and stored in
DatasetNode.column_lineage for every target dataset.
"""
from __future__ import annotations
import ast
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from datetime import datetime, timezone

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models import TransformationNode, DatasetNode, TransformationType, AnalysisTrace
from src.analyzers.sql_lineage import SQLLineageAnalyzer, extract_column_mappings
from src.analyzers.dag_config_parser import DAGConfigParser
from src.graph.knowledge_graph import KnowledgeGraph


class PythonDataFlowAnalyzer:
    """
    Detects data read/write operations in Python code:
    - pandas: read_csv, read_sql, read_parquet, to_csv, to_sql, to_parquet
    - SQLAlchemy: execute(), engine.connect()
    - PySpark: spark.read, df.write
    - Pathlib/open: file I/O
    """

    PANDAS_READS = {'read_csv', 'read_sql', 'read_sql_query', 'read_sql_table',
                    'read_parquet', 'read_json', 'read_excel', 'read_feather',
                    'read_orc', 'read_hdf', 'read_pickle'}
    PANDAS_WRITES = {'to_csv', 'to_sql', 'to_parquet', 'to_json', 'to_excel',
                     'to_feather', 'to_orc', 'to_hdf', 'to_pickle'}
    SPARK_READS  = {'csv', 'json', 'parquet', 'orc', 'text', 'jdbc', 'table', 'load'}
    SPARK_WRITES = {'csv', 'json', 'parquet', 'orc', 'text', 'jdbc', 'saveAsTable', 'save'}

    def analyze(self, path: str, source: str) -> Tuple[List[str], List[str]]:
        """Returns (source_datasets, target_datasets)."""
        sources: Set[str] = set()
        targets: Set[str] = set()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return self._analyze_regex(source)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            method_name = ""
            if isinstance(node.func, ast.Attribute):
                method_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                method_name = node.func.id
            if not method_name:
                continue
            dataset_name = self._extract_first_arg(node)
            if not dataset_name:
                continue
            if method_name in self.PANDAS_READS:
                sources.add(self._clean_dataset_name(dataset_name))
            elif method_name in self.PANDAS_WRITES:
                targets.add(self._clean_dataset_name(dataset_name))
            elif method_name in ('execute', 'run_query', 'query'):
                if any(kw in dataset_name.upper() for kw in ('SELECT', 'INSERT', 'UPDATE', 'FROM')):
                    tables = re.findall(r'\bFROM\s+(\w+)', dataset_name, re.IGNORECASE)
                    sources.update(t.lower() for t in tables)
            elif method_name == 'table':
                sources.add(self._clean_dataset_name(dataset_name))
        return list(sources), list(targets)

    def _extract_first_arg(self, call_node: ast.Call) -> str:
        if not call_node.args:
            return ""
        arg = call_node.args[0]
        try:
            val = ast.literal_eval(arg)
            return str(val)
        except Exception:
            if isinstance(arg, ast.Name):
                return f"<dynamic:{arg.id}>"
            if isinstance(arg, ast.JoinedStr):
                return "<dynamic:f-string>"
            return ""

    def _clean_dataset_name(self, name: str) -> str:
        if name.startswith('<dynamic'):
            return name
        name = os.path.basename(name)
        for ext in ('.csv', '.parquet', '.json', '.xlsx', '.orc', '.feather'):
            if name.endswith(ext):
                name = name[:-len(ext)]
        return name.lower().strip()

    def _analyze_regex(self, source: str) -> Tuple[List[str], List[str]]:
        read_pattern  = re.compile(r'pd\.read_\w+\([\'"]([^\'"]+)[\'"]', re.IGNORECASE)
        write_pattern = re.compile(r'\.to_\w+\([\'"]([^\'"]+)[\'"]', re.IGNORECASE)
        sources = [self._clean_dataset_name(m.group(1)) for m in read_pattern.finditer(source)]
        targets = [self._clean_dataset_name(m.group(1)) for m in write_pattern.finditer(source)]
        return sources, targets


# ── Column-lineage propagation ─────────────────────────────────────────────────

def _propagate_column_lineage(kg: KnowledgeGraph,
                               transform: TransformationNode) -> None:
    """
    Push column_mappings from a TransformationNode into the target DatasetNode(s).
    If the target dataset already has column_lineage, we merge (update).
    """
    if not transform.column_mappings:
        return
    for tgt_name in transform.target_datasets:
        if tgt_name in kg.datasets:
            existing = kg.datasets[tgt_name].column_lineage or {}
            existing.update(transform.column_mappings)
            kg.datasets[tgt_name].column_lineage = existing



def _infer_storage_type(dataset_name: str, source_file: str = "") -> str:
    """Infer a human-readable storage type from path and naming conventions."""
    name = dataset_name.lower()
    src  = source_file.lower().replace("\\", "/")
    # Raw / seed tables — check name first (independent of source file)
    if name.startswith("raw_") or name.startswith("seed_"):
        return "source table"
    if "seeds/" in src:
        return "seed (csv)"
    # dbt model files
    if "models/" in src:
        if name.startswith("stg_"):
            return "dbt staging"
        if name.startswith("int_"):
            return "dbt intermediate"
        if name.startswith("fct_") or name.startswith("fact_"):
            return "dbt fact"
        if name.startswith("dim_"):
            return "dbt dimension"
        return "dbt model"
    # Airflow / Spark
    if "spark" in src or "pyspark" in src:
        return "spark"
    if "airflow" in src or "dag" in src:
        return "airflow"
    # Notebook
    if ".ipynb" in src:
        return "notebook"
    # SQL but not dbt
    if ".sql" in src:
        return "sql view"
    # Python ETL
    if ".py" in src:
        return "python etl"
    return "unknown"

class Hydrologist:
    """
    Constructs the data lineage DAG by analysing all file types.
    After building the graph, column_lineage is propagated from each
    TransformationNode into its target DatasetNode(s) so that the
    KnowledgeGraph carries full column-level provenance.
    """

    def __init__(self, kg: KnowledgeGraph):
        self.kg         = kg
        self.py_flow    = PythonDataFlowAnalyzer()
        self.sql_lineage = SQLLineageAnalyzer()
        self.dag_parser  = DAGConfigParser()
        self.traces: List[AnalysisTrace] = []

    def _trace(self, action: str, target: str, result: str,
               source: str = "static_analysis"):
        self.traces.append(AnalysisTrace(
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent="hydrologist",
            action=action, target=target,
            result_summary=result, evidence_source=source,
        ))

    def run(self, repo_path: str) -> Dict[str, Any]:
        print(f"  [Hydrologist] Building data lineage DAG...")
        total_transforms = 0
        col_lineage_count = 0

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in
                       ('.git', '__pycache__', 'node_modules', '.venv',
                        'venv', '.cartography')]
            for fname in files:
                path = os.path.join(root, fname)
                suffix = Path(path).suffix.lower()
                try:
                    if suffix == '.py':
                        transforms = self._analyze_python(path)
                    elif suffix == '.sql':
                        transforms = self._analyze_sql(path)
                    elif suffix in ('.yml', '.yaml'):
                        transforms = self._analyze_yaml(path)
                    elif suffix == '.ipynb':
                        transforms = self._analyze_notebook(path)
                    else:
                        continue

                    for t in transforms:
                        self.kg.add_transformation(t)
                        # ── Infer storage_label on source + target datasets
                        for ds_name in list(t.source_datasets) + list(t.target_datasets):
                            if ds_name in self.kg.datasets:
                                ds_node = self.kg.datasets[ds_name]
                                if not getattr(ds_node, "storage_label", ""):
                                    ds_node.storage_label = _infer_storage_type(ds_name, t.source_file or "")
                        # ── Propagate column lineage into target DatasetNodes
                        if t.column_mappings:
                            _propagate_column_lineage(self.kg, t)
                            col_lineage_count += len(t.column_mappings)
                        total_transforms += 1

                except Exception as e:
                    self._trace("error", path, f"Failed: {e}")

        sources = self.kg.find_sources()
        sinks   = self.kg.find_sinks()
        self._trace("lineage_complete", repo_path,
                    f"Built lineage: {len(self.kg.datasets)} datasets, "
                    f"{total_transforms} transforms, "
                    f"{col_lineage_count} column-level mappings")

        print(f"  [Hydrologist] Found {len(self.kg.datasets)} datasets, "
              f"{total_transforms} transformations")
        if col_lineage_count:
            print(f"  [Hydrologist] Column-level lineage: {col_lineage_count} column mappings")
        print(f"  [Hydrologist] Sources (entry points): {sources[:5]}")
        print(f"  [Hydrologist] Sinks (outputs): {sinks[:5]}")

        return {
            "dataset_count":       len(self.kg.datasets),
            "transformation_count": total_transforms,
            "column_lineage_count": col_lineage_count,
            "sources":             sources,
            "sinks":               sinks,
        }

    # ── per-filetype analysis ──────────────────────────────────────────────────

    def _analyze_python(self, path: str) -> List[TransformationNode]:
        try:
            source = Path(path).read_text(encoding='utf-8', errors='replace')
        except Exception:
            return []
        dag_transforms = self.dag_parser.parse_file(path)
        if dag_transforms:
            return dag_transforms
        srcs, tgts = self.py_flow.analyze(path, source)
        if not srcs and not tgts:
            return []
        return [TransformationNode(
            name=f"py_flow_{Path(path).stem}",
            source_datasets=srcs,
            target_datasets=tgts,
            transformation_type=TransformationType.PANDAS.value,
            source_file=path,
        )]

    def _analyze_sql(self, path: str) -> List[TransformationNode]:
        repo_indicators = ['models/', 'analyses/', 'snapshots/']
        normalised = path.replace('\\', '/').replace('\\\\', '/')
        if any(indicator in normalised for indicator in repo_indicators):
            return self.sql_lineage.analyze_dbt_model(path)
        # For plain SQL files, also extract column-level lineage
        try:
            sql = Path(path).read_text(encoding='utf-8', errors='replace')
        except Exception:
            return []
        transforms = self.sql_lineage.extract_lineage(sql, path)
        # attach column lineage from SELECT analysis
        col_maps = extract_column_mappings(sql)
        for t in transforms:
            if col_maps and not t.column_mappings:
                t.column_mappings = col_maps
        return transforms

    def _analyze_yaml(self, path: str) -> List[TransformationNode]:
        return self.dag_parser.parse_file(path)

    def _analyze_notebook(self, path: str) -> List[TransformationNode]:
        import json
        try:
            nb = json.loads(Path(path).read_text(encoding='utf-8', errors='replace'))
            cells = []
            for cell in nb.get('cells', []):
                if cell.get('cell_type') == 'code':
                    cells.extend(cell.get('source', []))
            source = ''.join(cells)
            srcs, tgts = self.py_flow.analyze(path, source)
            if not srcs and not tgts:
                return []
            return [TransformationNode(
                name=f"notebook_{Path(path).stem}",
                source_datasets=srcs,
                target_datasets=tgts,
                transformation_type=TransformationType.PANDAS.value,
                source_file=path,
            )]
        except Exception:
            return []