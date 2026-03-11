"""
DAG config parser for Airflow DAG files and dbt schema.yml.
Extracts pipeline topology from configuration and Python DAG definitions.
"""
from __future__ import annotations
import ast
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models import TransformationNode, TransformationType

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class AirflowDAGParser:
    """
    Parses Airflow DAG Python files to extract:
    - Task dependencies (upstream >> downstream)
    - Data sources from operators
    - Pipeline topology
    """

    # Patterns for common Airflow operators
    OPERATOR_PATTERN = re.compile(
        r'(\w+)\s*=\s*(?:\w+\.)?(\w+Operator|PythonOperator|BashOperator|SQLExecuteQueryOperator|BigQueryOperator|SnowflakeOperator|SparkSubmitOperator)\s*\(',
        re.IGNORECASE
    )
    DEPENDENCY_PATTERN = re.compile(r'(\w+)\s*>>\s*(\w+)')
    DAG_ID_PATTERN = re.compile(r"dag_id\s*=\s*['\"]([^'\"]+)['\"]")
    SQL_FIELD = re.compile(r"sql\s*=\s*['\"]([^'\"]{5,})['\"]")
    TABLE_PATTERN = re.compile(r"(?:source_table|destination_table|table_name|source|destination)\s*=\s*['\"]([^'\"]+)['\"]")

    def parse_dag_file(self, path: str) -> List[TransformationNode]:
        try:
            source = Path(path).read_text(encoding='utf-8', errors='replace')
        except Exception:
            return []

        transformations = []
        dag_id_match = self.DAG_ID_PATTERN.search(source)
        dag_id = dag_id_match.group(1) if dag_id_match else os.path.basename(path)

        # Extract task operators
        tasks = {}
        for m in self.OPERATOR_PATTERN.finditer(source):
            task_var = m.group(1)
            operator_type = m.group(2)
            tasks[task_var] = operator_type

        # Extract dependencies
        deps = []
        for m in self.DEPENDENCY_PATTERN.finditer(source):
            deps.append((m.group(1), m.group(2)))

        # Extract data tables referenced
        sources = [m.group(1) for m in self.TABLE_PATTERN.finditer(source)]

        if tasks:
            node = TransformationNode(
                name=f"airflow_dag_{dag_id}",
                source_datasets=list(set(sources)),
                target_datasets=[],
                transformation_type=TransformationType.AIRFLOW_TASK.value,
                source_file=path,
            )
            transformations.append(node)

        # Try deeper AST analysis
        try:
            tree = ast.parse(source)
            transformations.extend(self._extract_from_ast(tree, path, dag_id))
        except Exception:
            pass

        return transformations

    def _extract_from_ast(self, tree: ast.AST, path: str, dag_id: str) -> List[TransformationNode]:
        """Extract tasks and their SQL/table references via AST."""
        transformations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    func_name = node.func.id

                if 'operator' in func_name.lower() or func_name.endswith('Hook'):
                    kwds = {kw.arg: kw for kw in node.keywords if kw.arg}
                    task_id = ""
                    if 'task_id' in kwds:
                        try:
                            task_id = ast.literal_eval(kwds['task_id'].value)
                        except Exception:
                            pass
        return transformations


class DBTSchemaParser:
    """
    Parses dbt schema.yml to extract model metadata, sources, and tests.
    """

    def parse_schema(self, path: str) -> Dict[str, Any]:
        result = {"models": [], "sources": [], "tests": []}
        if not YAML_AVAILABLE:
            return self._parse_schema_regex(path, result)
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                return result

            for model in data.get('models', []):
                if isinstance(model, dict):
                    result["models"].append({
                        "name": model.get("name", ""),
                        "description": model.get("description", ""),
                        "columns": [c.get("name","") for c in model.get("columns", []) if isinstance(c, dict)],
                    })

            for source in data.get('sources', []):
                if isinstance(source, dict):
                    for table in source.get('tables', []):
                        if isinstance(table, dict):
                            result["sources"].append({
                                "schema": source.get("name", ""),
                                "table": table.get("name", ""),
                                "description": table.get("description", ""),
                            })
        except Exception:
            return self._parse_schema_regex(path, result)
        return result

    def _parse_schema_regex(self, path: str, result: dict) -> dict:
        try:
            src = Path(path).read_text()
            names = re.findall(r'^\s*-\s*name:\s*(\S+)', src, re.MULTILINE)
            result["models"] = [{"name": n, "description": "", "columns": []} for n in names]
        except Exception:
            pass
        return result


class DAGConfigParser:
    """Unified parser — dispatches to Airflow or dbt parsers."""

    def __init__(self):
        self.airflow_parser = AirflowDAGParser()
        self.dbt_parser = DBTSchemaParser()

    def parse_file(self, path: str) -> List[TransformationNode]:
        fname = os.path.basename(path).lower()
        suffix = Path(path).suffix.lower()

        # dbt schema.yml
        if fname in ('schema.yml', 'schema.yaml', 'sources.yml', 'sources.yaml'):
            schema = self.dbt_parser.parse_schema(path)
            transforms = []
            for src in schema.get("sources", []):
                ds_name = f"{src['schema']}.{src['table']}"
                transforms.append(TransformationNode(
                    name=f"dbt_source_{ds_name}",
                    source_datasets=[ds_name],
                    target_datasets=[src['table']],
                    transformation_type=TransformationType.DBT_MODEL.value,
                    source_file=path,
                ))
            return transforms

        # Airflow Python DAG files
        if suffix == '.py' and any(kw in path for kw in ['dag', 'airflow', 'pipeline', 'flow']):
            return self.airflow_parser.parse_dag_file(path)

        # Generic Airflow YAML DAGs
        if suffix in ('.yml', '.yaml'):
            return self._parse_yaml_dag(path)

        return []

    def _parse_yaml_dag(self, path: str) -> List[TransformationNode]:
        if not YAML_AVAILABLE:
            return []
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                return []
            # Airflow YAML DAG format
            if 'dag_id' in data or 'tasks' in data:
                dag_id = data.get('dag_id', os.path.basename(path))
                tasks = data.get('tasks', {})
                sources = []
                if isinstance(tasks, dict):
                    for task_id, task_conf in tasks.items():
                        if isinstance(task_conf, dict):
                            for key in ('source_table', 'table_name', 'sql'):
                                if key in task_conf:
                                    sources.append(str(task_conf[key]))
                return [TransformationNode(
                    name=f"airflow_yaml_{dag_id}",
                    source_datasets=sources,
                    target_datasets=[],
                    transformation_type=TransformationType.AIRFLOW_TASK.value,
                    source_file=path,
                )]
        except Exception:
            pass
        return []
