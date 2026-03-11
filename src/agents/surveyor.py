"""
Agent 1: The Surveyor — Static Structure Analyst
Builds the module graph, computes PageRank, git velocity, dead code candidates.
"""
from __future__ import annotations
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime, timezone

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models import ModuleNode, FunctionNode, AnalysisTrace
from src.analyzers.tree_sitter_analyzer import TreeSitterAnalyzer
from src.graph.knowledge_graph import KnowledgeGraph


class Surveyor:
    """
    Performs deep static analysis:
    - Module import graph
    - Public API surface
    - Complexity signals
    - Git change velocity
    - Dead code candidates
    """

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.analyzer = TreeSitterAnalyzer()
        self.traces: List[AnalysisTrace] = []

    def _trace(self, action: str, target: str, result: str, source: str = "static_analysis"):
        self.traces.append(AnalysisTrace(
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent="surveyor",
            action=action,
            target=target,
            result_summary=result,
            evidence_source=source,
        ))

    def run(self, repo_path: str) -> Dict[str, Any]:
        print(f"  [Surveyor] Scanning files in {repo_path}...")
        modules, functions = self.analyzer.analyze_repo(repo_path)
        self._trace("scan_repo", repo_path, f"Found {len(modules)} modules, {len(functions)} functions")

        # Register all modules in knowledge graph
        for module in modules:
            self.kg.add_module(module)
        for fn in functions:
            self.kg.add_function(fn)

        print(f"  [Surveyor] Building import graph...")
        self._build_import_graph(repo_path, modules)

        print(f"  [Surveyor] Computing git velocity...")
        velocity_map = self.extract_git_velocity(repo_path, days=30)
        self._apply_git_velocity(velocity_map)

        print(f"  [Surveyor] Computing PageRank...")
        pagerank = self.kg.compute_pagerank()

        print(f"  [Surveyor] Finding circular dependencies...")
        cycles = self.kg.find_circular_dependencies()

        print(f"  [Surveyor] Detecting dead code candidates...")
        dead_code = self._detect_dead_code()

        high_velocity = sorted(
            velocity_map.items(), key=lambda x: x[1], reverse=True
        )[:20]

        self._trace("pagerank", repo_path, f"Top module: {max(pagerank, key=pagerank.get, default='none')}")
        self._trace("cycles", repo_path, f"Found {len(cycles)} circular dependency groups")

        return {
            "module_count": len(modules),
            "function_count": len(functions),
            "pagerank": pagerank,
            "circular_deps": cycles,
            "dead_code_candidates": dead_code,
            "high_velocity_files": high_velocity,
            "errors": self.analyzer.errors,
        }

    def _build_import_graph(self, repo_path: str, modules: List[ModuleNode]):
        """Resolve import statements to actual file paths and add edges."""
        # Build a lookup: module_name -> file_path
        path_index: Dict[str, str] = {}
        for m in modules:
            rel = os.path.relpath(m.path, repo_path)
            # Convert path to module notation
            mod_name = rel.replace(os.sep, '.').replace('/', '.').rstrip('.py')
            if mod_name.endswith('.py'):
                mod_name = mod_name[:-3]
            path_index[mod_name] = m.path
            # Also index by basename
            basename = Path(m.path).stem
            if basename not in path_index:
                path_index[basename] = m.path

        for m in modules:
            for imp in m.imports:
                # Try to resolve import to a file in the repo
                target_path = self._resolve_import(imp, path_index, repo_path)
                if target_path and target_path != m.path:
                    self.kg.add_import_edge(m.path, target_path)

    def _resolve_import(self, import_name: str, path_index: Dict[str, str], repo_path: str) -> str:
        """Try to resolve an import name to a file path."""
        # Direct match
        if import_name in path_index:
            return path_index[import_name]
        # Partial match (last component)
        parts = import_name.split('.')
        for i in range(len(parts), 0, -1):
            key = '.'.join(parts[:i])
            if key in path_index:
                return path_index[key]
        # Try basename only
        leaf = parts[-1]
        if leaf in path_index:
            return path_index[leaf]
        return ""

    def extract_git_velocity(self, repo_path: str, days: int = 30) -> Dict[str, int]:
        """Parse git log to compute change frequency per file."""
        velocity: Dict[str, int] = {}
        try:
            result = subprocess.run(
                ['git', 'log', f'--since={days} days ago', '--name-only', '--pretty=format:'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if line and not line.startswith('commit'):
                        full_path = os.path.join(repo_path, line)
                        velocity[full_path] = velocity.get(full_path, 0) + 1
        except Exception as e:
            self._trace("git_velocity", repo_path, f"Git log failed: {e}", "git_log")
        return velocity

    def _apply_git_velocity(self, velocity_map: Dict[str, int]):
        for path, count in velocity_map.items():
            if path in self.kg.modules:
                self.kg.modules[path].change_velocity_30d = count

    def _detect_dead_code(self) -> List[str]:
        """
        Dead code candidates: modules with no incoming import edges
        and no exports referenced by other modules.
        """
        dead = []
        if len(self.kg.module_graph) == 0:
            return dead
        for node in self.kg.module_graph.nodes:
            in_degree = self.kg.module_graph.in_degree(node)
            if in_degree == 0 and node in self.kg.modules:
                m = self.kg.modules[node]
                # Skip entry points and config files
                basename = os.path.basename(node).lower()
                if basename in ('main.py', 'cli.py', 'app.py', 'manage.py',
                                 'setup.py', 'conftest.py', '__init__.py'):
                    continue
                if any(skip in basename for skip in ('test_', '_test', 'setup', 'conftest')):
                    continue
                m.is_dead_code_candidate = True
                dead.append(node)
        return dead
