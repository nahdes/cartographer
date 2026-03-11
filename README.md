# 🗺️ Brownfield Cartographer

**Codebase Intelligence System for Rapid FDE Onboarding**

> "A cartographer does not need to walk every road to produce a map — they build systematic methods for extracting structure and representing it."

## What it does

The Brownfield Cartographer ingests any local path (or GitHub URL) and produces a **living, queryable knowledge graph** of the codebase's architecture, data flows, and semantic structure.

### Outputs

| Artifact                  | Description                                                 |
| ------------------------- | ----------------------------------------------------------- |
| `CODEBASE.md`             | Living context file — inject directly into AI coding agents |
| `onboarding_brief.md`     | FDE Day-One Brief answering the 5 critical questions        |
| `module_graph.json`       | Import dependency graph (NetworkX) with PageRank scores     |
| `lineage_graph.json`      | Data lineage DAG (sources → transforms → sinks)             |
| `cartography_trace.jsonl` | Full audit log of every analysis action                     |

## Installation

```bash
# Requires Python 3.11+
pip install networkx scikit-learn numpy pyyaml
# Optional for LLM-powered analysis:
pip install anthropic
# Optional for better SQL parsing:
pip install sqlglot
```

## Usage

### Analyze a codebase

```bash
# Local path
python src/cli.py analyze /path/to/your/repo

# GitHub URL (clones automatically)
python src/cli.py analyze https://github.com/dbt-labs/jaffle_shop

# With LLM-powered analysis (Anthropic API)
python src/cli.py analyze /path/to/repo --api-key sk-ant-...

# Specify output directory
python src/cli.py analyze /path/to/repo --output-dir /tmp/cartography

# Incremental update (only re-analyze changed files)
python src/cli.py analyze /path/to/repo --incremental
```

### Query the knowledge graph

```bash
# Interactive query mode
python src/cli.py query /path/to/repo --interactive

# Single query
python src/cli.py query /path/to/repo --query "where is the revenue calculation?"

# Blast radius
python src/cli.py blast-radius /path/to/repo src/transforms/revenue_calc.py

# Lineage trace
python src/cli.py lineage /path/to/repo daily_active_users --direction upstream
```

### Interactive Navigator commands

```
Navigator> find revenue calculation
Navigator> lineage daily_active_users upstream
Navigator> blast src/transforms/revenue_calc.py
Navigator> explain src/ingestion/kafka_consumer.py
Navigator> quit
```

## Architecture

```
CLI (cli.py)
    └── Orchestrator (orchestrator.py)
            ├── Agent 1: Surveyor  (static structure)
            │       └── TreeSitterAnalyzer (Python AST, SQL regex, YAML)
            ├── Agent 2: Hydrologist (data lineage)
            │       ├── PythonDataFlowAnalyzer (pandas/spark read/write)
            │       ├── SQLLineageAnalyzer (sqlglot or regex)
            │       └── DAGConfigParser (Airflow, dbt schema.yml)
            ├── Agent 3: Semanticist (LLM analysis)
            │       ├── Purpose statement generation
            │       ├── Documentation drift detection
            │       ├── Domain clustering (TF-IDF + KMeans)
            │       └── FDE Day-One Question synthesis
            └── Agent 4: Archivist (artifact generation)
                    ├── CODEBASE.md
                    ├── onboarding_brief.md
                    └── cartography_trace.jsonl

Knowledge Graph (KnowledgeGraph)
    ├── module_graph: NetworkX DiGraph (import edges, PageRank)
    └── lineage_graph: NetworkX DiGraph (CONSUMES/PRODUCES edges)
```

## Supported Languages & Patterns

| Language          | Parser          | Extracts                                      |
| ----------------- | --------------- | --------------------------------------------- |
| Python            | stdlib `ast`    | imports, functions, classes, read/write calls |
| SQL               | sqlglot + regex | table dependencies, CTEs, dbt ref()           |
| dbt models        | sqlglot + regex | full DAG with ref() and source()              |
| Airflow DAGs      | ast + regex     | task dependencies, pipeline topology          |
| YAML              | PyYAML          | dbt schema.yml, Airflow YAML DAGs             |
| Jupyter Notebooks | JSON + ast      | data reads/writes from code cells             |

## LLM Integration

Set `ANTHROPIC_API_KEY` environment variable for:

- **Purpose statements** grounded in actual code (not docstrings)
- **Documentation drift detection** — flags where docstring ≠ implementation
- **FDE Day-One answers** synthesized from full architectural context
- **Module explanations** via `explain_module()` Navigator tool

Without an API key, the system uses heuristic analysis (path-based domain inference, AST-based purpose extraction).

## The Five FDE Day-One Questions

Every `onboarding_brief.md` answers:

1. **What is the primary data ingestion path?** (Source nodes with in-degree=0)
2. **What are the 3-5 most critical output datasets?** (Sink nodes + PageRank)
3. **What is the blast radius if the critical module fails?** (BFS from PageRank hub)
4. **Where is the business logic concentrated vs. distributed?** (Domain clusters)
5. **What has changed most frequently in the last 90 days?** (Git velocity map)
