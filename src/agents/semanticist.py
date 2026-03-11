"""
Agent 3: The Semanticist — LLM-Powered Purpose Analyst
Generates purpose statements, detects doc drift, clusters domains via
TF-IDF+KMeans, maintains a vector semantic_index/, answers Day-One questions.
"""
from __future__ import annotations
import os, json, re, pickle, math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models import ModuleNode, AnalysisTrace
from src.graph.knowledge_graph import KnowledgeGraph

try:
    import anthropic
    ANTHROPIC_OK = True
except ImportError:
    ANTHROPIC_OK = False

try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


# ── Context budget ────────────────────────────────────────────────────────────

class ContextWindowBudget:
    def __init__(self, max_tokens: int = 200_000):
        self.max_tokens = max_tokens
        self.used = 0
        self.calls = 0

    def can_afford(self, text: str, reserve: int = 1_000) -> bool:
        return self.used + len(text) // 4 + reserve < self.max_tokens

    def consume(self, inp: int, out: int):
        self.used += inp + out; self.calls += 1

    def summary(self) -> str:
        return f"~{self.used:,} tokens across {self.calls} API calls"


# ── Vector semantic index ─────────────────────────────────────────────────────

class SemanticIndex:
    """
    TF-IDF vector store for module purpose statements.
    Persisted to semantic_index/ as a pickle so it survives between runs.
    Provides cosine-similarity search: find_similar(query, k=5).
    """

    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._pkl = self.index_dir / "tfidf_index.pkl"
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None        # sparse TF-IDF matrix
        self.paths:  List[str] = []
        self.texts:  List[str] = []
        self._loaded = False

    def build(self, modules: List[ModuleNode]) -> None:
        if not SKLEARN_OK or len(modules) < 2:
            return
        self.paths = [m.path for m in modules]
        self.texts = [
            f"{m.purpose_statement} {m.path} {' '.join(m.exports)} {m.domain_cluster}"
            for m in modules
        ]
        self.vectorizer = TfidfVectorizer(
            max_features=1000, stop_words="english",
            ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(self.texts)
        self._loaded = True
        self._save()

    def find_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self._loaded:
            self._load()
        if not self._loaded or self.vectorizer is None:
            return []
        try:
            qvec = self.vectorizer.transform([query])
            sims = cosine_similarity(qvec, self.matrix).flatten()
            top_k = sims.argsort()[::-1][:k]
            return [{"path": self.paths[i], "score": float(sims[i]),
                     "text_snippet": self.texts[i][:120]}
                    for i in top_k if sims[i] > 0.01]
        except Exception:
            return []

    def _save(self):
        try:
            with open(self._pkl, "wb") as f:
                pickle.dump({"vectorizer": self.vectorizer,
                             "matrix": self.matrix,
                             "paths": self.paths,
                             "texts": self.texts}, f)
        except Exception:
            pass

    def _load(self):
        if not self._pkl.exists():
            return
        try:
            with open(self._pkl, "rb") as f:
                d = pickle.load(f)
            self.vectorizer = d["vectorizer"]
            self.matrix     = d["matrix"]
            self.paths      = d["paths"]
            self.texts      = d["texts"]
            self._loaded    = True
        except Exception:
            pass


# ── Semanticist ───────────────────────────────────────────────────────────────

class Semanticist:
    DOMAIN_LABELS = [
        "ingestion", "transformation", "serving", "monitoring",
        "orchestration", "utilities", "testing", "configuration",
    ]
    DOMAIN_KW = {
        "ingestion":      ["ingest","extract","load","source","reader","fetch","pull","kafka","s3"],
        "transformation": ["transform","clean","process","enrich","normalize","compute","calc"],
        "serving":        ["serve","api","endpoint","export","output","publish","graphql","rest"],
        "monitoring":     ["monitor","alert","metric","health","check","log","sla","anomaly"],
        "orchestration":  ["dag","pipeline","flow","schedule","airflow","prefect","luigi","step"],
        "utilities":      ["util","helper","common","shared","base","mixin","decorator"],
        "testing":        ["test","spec","fixture","mock","assert","pytest","unittest"],
        "configuration":  ["config","setting","env","secret","const","param","profile"],
    }

    def __init__(self, kg: KnowledgeGraph, api_key: str = "",
                 output_dir: str = ""):
        self.kg = kg
        self.budget = ContextWindowBudget()
        self.traces: List[AnalysisTrace] = []
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.client = None
        self.llm_ok = False
        self.semantic_index: Optional[SemanticIndex] = None
        if output_dir:
            self.semantic_index = SemanticIndex(
                os.path.join(output_dir, "semantic_index"))
        if ANTHROPIC_OK and self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.llm_ok = True
        else:
            print("  [Semanticist] No API key — using heuristic analysis")

    def _trace(self, action, target, result, source="llm_inference"):
        self.traces.append(AnalysisTrace(
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent="semanticist", action=action, target=target,
            result_summary=result, evidence_source=source))

    def _llm(self, prompt: str, max_tokens: int = 500) -> str:
        if not self.llm_ok or not self.budget.can_afford(prompt):
            return ""
        try:
            resp = self.client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}])
            out = resp.content[0].text if resp.content else ""
            self.budget.consume(
                getattr(resp.usage, "input_tokens", len(prompt)//4),
                getattr(resp.usage, "output_tokens", len(out)//4))
            return out
        except Exception as e:
            return f"[LLM ERROR: {e}]"

    # ── Purpose statements ────────────────────────────────────────────────────

    def generate_purpose(self, module: ModuleNode) -> str:
        try:
            source = Path(module.path).read_text(encoding="utf-8", errors="replace")[:3000]
        except Exception:
            return self._heuristic_purpose(module)
        if self.llm_ok:
            prompt = (
                f"Analyze this module and write a 2-3 sentence purpose statement.\n"
                f"Focus on BUSINESS FUNCTION, NOT implementation details.\n"
                f"Do NOT repeat the docstring — derive meaning from actual code.\n\n"
                f"File: {module.path}\n```\n{source}\n```\n\nRespond with ONLY the purpose statement.")
            result = self._llm(prompt, max_tokens=150)
            if result and not result.startswith("["):
                return result.strip()
        return self._heuristic_purpose(module)

    def _heuristic_purpose(self, module: ModuleNode) -> str:
        p = module.path.lower()
        hints = []
        for domain, kws in self.DOMAIN_KW.items():
            if any(kw in p for kw in kws):
                hints.append(domain); break
        domain = hints[0] if hints else "general purpose"
        exports_str = ", ".join(module.exports[:5]) or "no exports"
        return (f"Module responsible for {domain} in the data pipeline. "
                f"Exports: {exports_str}. "
                f"Complexity: {module.complexity_score:.0f} branches.")

    # ── Documentation drift ───────────────────────────────────────────────────

    def detect_doc_drift(self, module: ModuleNode, purpose: str) -> bool:
        if not module.docstring or not purpose:
            return False
        doc_w   = set(re.findall(r"\b\w{5,}\b", module.docstring.lower()))
        purp_w  = set(re.findall(r"\b\w{5,}\b", purpose.lower()))
        if not doc_w:
            return False
        overlap = len(doc_w & purp_w) / len(doc_w)
        return overlap < 0.10   # <10% word overlap = likely stale docstring

    # ── Domain clustering ─────────────────────────────────────────────────────

    def cluster_domains(self) -> Dict[str, List[str]]:
        modules = list(self.kg.modules.values())
        if not modules:
            return {}
        if SKLEARN_OK and len(modules) >= 3:
            return self._cluster_sklearn(modules)
        return self._cluster_heuristic(modules)

    def _cluster_sklearn(self, modules):
        texts = [f"{m.purpose_statement} {m.path} {' '.join(m.exports)}" for m in modules]
        paths = [m.path for m in modules]
        try:
            k = min(len(self.DOMAIN_LABELS), len(modules))
            vec = TfidfVectorizer(max_features=300, stop_words="english", sublinear_tf=True)
            X   = vec.fit_transform(texts)
            km  = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            clusters: Dict[int, List[str]] = defaultdict(list)
            for path, label in zip(paths, labels):
                clusters[label].append(path)
            named: Dict[str, List[str]] = {}
            for cid, cpaths in clusters.items():
                name = self._infer_cluster_name(cpaths)
                named[name] = cpaths
                for p in cpaths:
                    if p in self.kg.modules:
                        self.kg.modules[p].domain_cluster = name
            return named
        except Exception:
            return self._cluster_heuristic(modules)

    def _cluster_heuristic(self, modules):
        clusters: Dict[str, List[str]] = {k: [] for k in self.DOMAIN_LABELS}
        clusters["other"] = []
        for m in modules:
            p = m.path.lower()
            assigned = False
            for domain, kws in self.DOMAIN_KW.items():
                if any(kw in p for kw in kws):
                    clusters[domain].append(m.path)
                    m.domain_cluster = domain
                    assigned = True; break
            if not assigned:
                clusters["other"].append(m.path)
                m.domain_cluster = "other"
        return {k: v for k, v in clusters.items() if v}

    def _infer_cluster_name(self, paths):
        counts: Dict[str, int] = defaultdict(int)
        for path in paths:
            p = path.lower()
            for domain, kws in self.DOMAIN_KW.items():
                if any(kw in p for kw in kws):
                    counts[domain] += 1
        return max(counts, key=counts.get) if counts else "other"

    # ── Day-One answers ───────────────────────────────────────────────────────

    def answer_day_one(self, surveyor_result: Dict,
                       hydro_result: Dict) -> Dict[str, str]:
        sources      = hydro_result.get("sources", [])
        sinks        = hydro_result.get("sinks", [])
        high_vel     = surveyor_result.get("high_velocity_files", [])
        cycles       = surveyor_result.get("circular_deps", [])
        pagerank     = surveyor_result.get("pagerank", {})
        top_modules  = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
        if self.llm_ok:
            return self._llm_day_one(sources, sinks, high_vel, top_modules, cycles)
        return self._heuristic_day_one(sources, sinks, high_vel, top_modules, cycles)

    def _heuristic_day_one(self, sources, sinks, high_vel, top_modules, cycles):
        src_str  = ", ".join(sources[:5]) or "Not detected"
        sink_str = ", ".join(sinks[:5])   or "Not detected"
        top_str  = ", ".join(Path(p).name for p, _ in top_modules[:3]) or "Unknown"
        hv_str   = ", ".join(Path(p).name for p, _ in high_vel[:3])   or "No git history"
        cyc_str  = f"{len(cycles)} circular dependency group(s)" if cycles else "None"
        return {
            "q1_primary_ingestion":  f"Entry points (in-degree=0 datasets): {src_str}",
            "q2_critical_outputs":   f"Output sinks (out-degree=0 datasets): {sink_str}",
            "q3_blast_radius":       f"Highest-impact modules by PageRank: {top_str}. Run blast_radius() for full impact.",
            "q4_business_logic":     f"Logic concentrated in: {top_str}. Circular deps: {cyc_str}.",
            "q5_change_velocity":    f"Highest-velocity files (30d): {hv_str}. Active pain points.",
        }

    def _llm_day_one(self, sources, sinks, high_vel, top_modules, cycles):
        ctx = (f"Data Sources: {sources[:10]}\nData Sinks: {sinks[:10]}\n"
               f"High-velocity files: {[p for p, _ in high_vel[:10]]}\n"
               f"PageRank hubs: {[p for p, _ in top_modules]}\n"
               f"Circular deps: {cycles[:3]}")
        prompt = (
            f"You are analyzing a production data engineering codebase.\n{ctx}\n\n"
            "Answer the Five FDE Day-One Questions with specific evidence (file paths, dataset names):\n"
            "1. Primary data ingestion path?\n2. 3-5 critical output datasets?\n"
            "3. Blast radius if the most critical module fails?\n"
            "4. Where is business logic concentrated vs. distributed?\n"
            "5. Change velocity map (git activity last 90 days)?\n\n"
            "Respond as JSON with keys: q1_primary_ingestion, q2_critical_outputs, "
            "q3_blast_radius, q4_business_logic, q5_change_velocity")
        raw = self._llm(prompt, max_tokens=800)
        try:
            clean = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
            return json.loads(clean)
        except Exception:
            return self._heuristic_day_one(sources, sinks, high_vel, top_modules, [])

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self, surveyor_result: Dict, hydro_result: Dict,
            output_dir: str = "") -> Dict[str, Any]:
        modules = list(self.kg.modules.values())
        pagerank = surveyor_result.get("pagerank", {})
        modules_sorted = sorted(modules,
                                key=lambda m: pagerank.get(m.path, 0), reverse=True)

        print(f"  [Semanticist] Generating purpose statements ({len(modules)} modules)...")
        for m in modules_sorted:
            purpose = self.generate_purpose(m)
            m.purpose_statement = purpose
            m.doc_drift_flag = self.detect_doc_drift(m, purpose)
            if m.doc_drift_flag:
                self._trace("doc_drift", m.path, "Docstring contradicts implementation",
                            "static_analysis")

        print("  [Semanticist] Clustering modules into domains...")
        domain_clusters = self.cluster_domains()

        print("  [Semanticist] Building semantic index (semantic_index/)...")
        if self.semantic_index:
            self.semantic_index.build(modules_sorted)
        elif output_dir and SKLEARN_OK:
            self.semantic_index = SemanticIndex(
                os.path.join(output_dir, "semantic_index"))
            self.semantic_index.build(modules_sorted)

        print("  [Semanticist] Answering Day-One Questions...")
        day_one = self.answer_day_one(surveyor_result, hydro_result)

        self._trace("summary", "repo",
                    f"Processed {len(modules)} modules. {self.budget.summary()}",
                    "static_analysis")

        return {
            "domain_clusters":  domain_clusters,
            "day_one_answers":  day_one,
            "doc_drift_count":  sum(1 for m in modules if m.doc_drift_flag),
            "budget_summary":   self.budget.summary(),
            "semantic_index_built": self.semantic_index is not None and self.semantic_index._loaded,
        }
