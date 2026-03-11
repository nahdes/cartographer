"""
Orchestrator — wires all agents in sequence:
Surveyor → Hydrologist → Semanticist → Archivist
"""
from __future__ import annotations
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import CartographyResult
from src.graph.knowledge_graph import KnowledgeGraph
from src.agents.surveyor import Surveyor
from src.agents.hydrologist import Hydrologist
from src.agents.semanticist import Semanticist
from src.agents.archivist import Archivist
from src.agents.navigator import Navigator


class Orchestrator:
    """Runs the full cartography pipeline."""

    def __init__(self, repo_path: str, output_dir: Optional[str] = None, api_key: str = ""):
        self.repo_path = os.path.abspath(repo_path)
        self.repo_name = Path(repo_path).name
        self.output_dir = output_dir or os.path.join(os.getcwd(), ".cartography")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.kg = KnowledgeGraph()
        self.result: Optional[CartographyResult] = None

    def clone_if_url(self, url_or_path: str) -> str:
        """If given a GitHub URL, clone it to a temp dir and return the path."""
        if url_or_path.startswith("http") or url_or_path.startswith("git@"):
            print(f"  [Orchestrator] Cloning {url_or_path}...")
            tmp_dir = tempfile.mkdtemp(prefix="cartographer_")
            result = subprocess.run(
                ['git', 'clone', '--depth=50', url_or_path, tmp_dir],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                raise RuntimeError(f"Git clone failed: {result.stderr}")
            return tmp_dir
        return url_or_path

    def run(self, incremental: bool = False) -> CartographyResult:
        ts = datetime.now(timezone.utc).isoformat()
        print(f"\n🗺️  Brownfield Cartographer — {self.repo_name}")
        print(f"   Path: {self.repo_path}")
        print(f"   Output: {self.output_dir}")
        print()

        # Check for incremental mode
        if incremental and self._has_previous_run():
            print("  [Orchestrator] Incremental mode: analyzing only changed files...")
            changed = self._get_changed_files()
            if not changed:
                print("  [Orchestrator] No changes detected since last run.")
                return self._load_previous_result()
        # ── Phase 1: Surveyor ────────────────────────────────────────────────
        print("📐 Phase 1: Surveyor (Static Structure Analysis)")
        surveyor = Surveyor(self.kg)
        surveyor_result = surveyor.run(self.repo_path)

        # ── Phase 2: Hydrologist ─────────────────────────────────────────────
        print("\n💧 Phase 2: Hydrologist (Data Lineage Analysis)")
        hydrologist = Hydrologist(self.kg)
        hydro_result = hydrologist.run(self.repo_path)

        # ── Phase 3: Semanticist ─────────────────────────────────────────────
        print("\n🧠 Phase 3: Semanticist (LLM-Powered Analysis)")
        semanticist = Semanticist(self.kg, api_key=self.api_key)
        semantic_result = semanticist.run(surveyor_result, hydro_result)

        # ── Assemble CartographyResult ───────────────────────────────────────
        self.result = CartographyResult(
            repo_path=self.repo_path,
            repo_name=self.repo_name,
            analysis_timestamp=ts,
            module_nodes=self.kg.modules,
            dataset_nodes=self.kg.datasets,
            function_nodes=self.kg.functions,
            transformation_nodes=self.kg.transformations,
            domain_clusters=semantic_result.get("domain_clusters", {}),
            day_one_answers=semantic_result.get("day_one_answers", {}),
            high_velocity_files=surveyor_result.get("high_velocity_files", []),
            circular_dependencies=surveyor_result.get("circular_deps", []),
            pagerank_scores=surveyor_result.get("pagerank", {}),
            errors=surveyor_result.get("errors", []),
        )

        # ── Phase 4: Archivist ───────────────────────────────────────────────
        print("\n📚 Phase 4: Archivist (Writing Artifacts)")
        all_traces = surveyor.traces + hydrologist.traces + semanticist.traces
        archivist = Archivist(self.kg, self.output_dir)
        archivist.run(self.result, all_traces)

        # Save full result JSON for incremental mode
        result_path = os.path.join(self.output_dir, "cartography_result.json")
        with open(result_path, 'w') as f:
            json.dump(self.result.to_dict(), f, indent=2, default=str)

        print(f"\n✅ Cartography complete!")
        print(f"   📁 Artifacts: {self.output_dir}/")
        print(f"   📋 CODEBASE.md — inject into AI agents")
        print(f"   📝 onboarding_brief.md — FDE Day-One Brief")
        print(f"   🔗 module_graph.json — import dependency graph")
        print(f"   🌊 lineage_graph.json — data flow DAG")
        print(f"   📊 cartography_trace.jsonl — audit log")

        self._print_summary(surveyor_result, hydro_result, semantic_result)
        return self.result

    def get_navigator(self) -> Navigator:
        if not self.result:
            raise RuntimeError("Run analyze() before querying")
        return Navigator(self.kg, self.result, self.api_key)

    def _print_summary(self, surveyor, hydro, semantic):
        print(f"\n📊 Summary:")
        print(f"   Modules analyzed: {surveyor.get('module_count', 0)}")
        print(f"   Functions found:  {surveyor.get('function_count', 0)}")
        print(f"   Datasets tracked: {hydro.get('dataset_count', 0)}")
        print(f"   Transformations:  {hydro.get('transformation_count', 0)}")
        print(f"   Circular deps:    {len(surveyor.get('circular_deps', []))}")
        print(f"   Doc drift flags:  {semantic.get('doc_drift_count', 0)}")
        print(f"   LLM budget:       {semantic.get('budget_summary', 'N/A')}")

    def _has_previous_run(self) -> bool:
        return os.path.exists(os.path.join(self.output_dir, "cartography_result.json"))

    def _get_changed_files(self) -> list:
        try:
            result = subprocess.run(
                ['git', 'diff', '--name-only', 'HEAD~1', 'HEAD'],
                cwd=self.repo_path, capture_output=True, text=True, timeout=15
            )
            return result.stdout.splitlines() if result.returncode == 0 else []
        except Exception:
            return []

    def _load_previous_result(self) -> CartographyResult:
        result_path = os.path.join(self.output_dir, "cartography_result.json")
        with open(result_path) as f:
            data = json.load(f)
        # Reconstruct minimal result
        from src.models import ModuleNode, DatasetNode, FunctionNode, TransformationNode
        r = CartographyResult(
            repo_path=data["repo_path"],
            repo_name=data["repo_name"],
            analysis_timestamp=data["analysis_timestamp"],
        )
        return r