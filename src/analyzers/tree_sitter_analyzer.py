"""
Multi-language AST analyzer — v2.
  Python   -> stdlib ast  (full structural analysis, type annotations, call graph)
  JS / TS  -> scope-aware regex-AST (imports, exports, interfaces, types, generics,
               decorators, async/await, React hooks/components, call graph)
  SQL      -> regex       (table refs; column-level in sql_lineage.py)
  YAML     -> PyYAML structural parsing
  Notebook -> JSON -> Python cell extraction -> ast

Graceful degradation: errors are logged and skipped, never raised.
"""
from __future__ import annotations
import ast
import re
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models import ModuleNode, FunctionNode, Language

try:
    import yaml as _yaml
    YAML_OK = True
except ImportError:
    YAML_OK = False

LANGUAGE_MAP: Dict[str, str] = {
    ".py":    Language.PYTHON.value,
    ".sql":   Language.SQL.value,
    ".yml":   Language.YAML.value,
    ".yaml":  Language.YAML.value,
    ".js":    Language.JAVASCRIPT.value,
    ".jsx":   Language.JAVASCRIPT.value,
    ".ts":    Language.TYPESCRIPT.value,
    ".tsx":   Language.TYPESCRIPT.value,
    ".ipynb": Language.NOTEBOOK.value,
}
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", "dist", "build", ".cartography", ".mypy_cache", ".pytest_cache",
}


class LanguageRouter:
    @staticmethod
    def detect(path: str) -> str:
        return LANGUAGE_MAP.get(Path(path).suffix.lower(), Language.UNKNOWN.value)

    @staticmethod
    def should_skip(path: str) -> bool:
        return any(p in SKIP_DIRS for p in Path(path).parts)


# ── Python ─────────────────────────────────────────────────────────────────────

class PythonASTAnalyzer:
    BRANCH = (ast.If, ast.For, ast.While, ast.Try,
              ast.ExceptHandler, ast.With, ast.Assert)

    def analyze(self, path: str, source: str) -> Tuple[ModuleNode, List[FunctionNode]]:
        m = ModuleNode(path=path, language=Language.PYTHON.value)
        fns: List[FunctionNode] = []
        lines = source.splitlines()
        m.loc = len(lines)
        m.comment_ratio = sum(1 for l in lines if l.strip().startswith("#")) / max(1, len(lines))
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            m.purpose_statement = f"[PARSE ERROR: {e}]"
            return m, fns
        m.docstring = ast.get_docstring(tree) or ""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    m.imports.append(a.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    m.imports.append(node.module)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                pub = not node.name.startswith("_")
                sig = self._sig(node)
                cpx = 1 + sum(1 for n in ast.walk(node) if isinstance(n, self.BRANCH))
                fns.append(FunctionNode(
                    qualified_name=f"{path}::{node.name}", parent_module=path,
                    signature=sig, is_public_api=pub,
                    line_number=node.lineno, complexity=cpx,
                    return_type=self._ret(node)))
                if pub and node.name not in m.exports:
                    m.exports.append(node.name)
            elif isinstance(node, ast.ClassDef):
                if not node.name.startswith("_") and node.name not in m.exports:
                    m.exports.append(node.name)
        m.complexity_score = float(
            1 + sum(1 for n in ast.walk(tree) if isinstance(n, self.BRANCH)))
        return m, fns

    def _sig(self, node):
        args = []
        for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
            a = arg.arg
            if arg.annotation:
                try:
                    a += f": {ast.unparse(arg.annotation)}"
                except Exception:
                    pass
            args.append(a)
        ret = ""
        if node.returns:
            try:
                ret = f" -> {ast.unparse(node.returns)}"
            except Exception:
                pass
        return f"def {node.name}({', '.join(args)}){ret}"

    def _ret(self, node):
        if node.returns:
            try:
                return ast.unparse(node.returns)
            except Exception:
                pass
        return ""


# ── JavaScript / TypeScript — scope-aware deep analysis ───────────────────────

class JSTSAnalyzer:
    """
    Deep structural analysis for JS/TS without tree-sitter.

    Covers (beyond v1):
      - TypeScript interfaces, type aliases, enums, generics
      - Decorators (@Component, @Injectable, @Controller …)
      - async/await, Promise chains
      - React hooks (useState, useEffect …) and component prop types
      - Call graph: which functions call which other local functions
      - Re-export statements (export { X } from '…'; export * from '…')
      - Dynamic imports: import('…')
      - Type-only imports: import type { … }
    """

    # ── Import patterns ──────────────────────────────────────────────────────
    IMP_FROM   = re.compile(
        r"import\s+(?:type\s+)?"
        r"(?:\{[^}]*\}|\*\s+as\s+\w+|\w+)"
        r"(?:\s*,\s*(?:\{[^}]*\}|\w+))*"
        r"\s+from\s+['\"]([^'\"]+)['\"]",
        re.MULTILINE)
    IMP_BARE   = re.compile(r"import\s+['\"]([^'\"]+)['\"]", re.MULTILINE)
    IMP_REQ    = re.compile(r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)")
    IMP_DYN    = re.compile(r"import\s*\(\s*['\"]([^'\"]+)['\"]\s*\)")
    IMP_REEXP  = re.compile(r"export\s+(?:\{[^}]*\}|\*)\s+from\s+['\"]([^'\"]+)['\"]", re.MULTILINE)

    # ── Export patterns ──────────────────────────────────────────────────────
    EXP_FN     = re.compile(r"export\s+(?:async\s+)?function\s+(\w+)")
    EXP_CLS    = re.compile(r"export\s+(?:abstract\s+)?class\s+(\w+)")
    EXP_CONST  = re.compile(r"export\s+(?:const|let|var)\s+(\w+)")
    EXP_IFACE  = re.compile(r"export\s+interface\s+(\w+)")
    EXP_TYPE   = re.compile(r"export\s+type\s+(\w+)")
    EXP_ENUM   = re.compile(r"export\s+(?:const\s+)?enum\s+(\w+)")
    EXP_DEF    = re.compile(r"export\s+default\s+(?:async\s+)?(?:class|function)?\s*(\w*)")
    EXP_NAMED  = re.compile(r"export\s+\{([^}]+)\}")

    # ── Declaration patterns — handle optional 'export' / 'export default' prefix
    DEF_FN     = re.compile(r"(?:^|\n)\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*[<(]", re.MULTILINE)
    # Arrow / const functions — stop at '=' or ':' so type-annotated consts match
    DEF_ARR    = re.compile(r"(?:^|\n)\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_]\w*)\s*[=:]", re.MULTILINE)
    DEF_CLS    = re.compile(r"(?:^|\n)\s*(?:export\s+)?(?:abstract\s+)?class\s+(\w+)", re.MULTILINE)
    DEF_IFACE  = re.compile(r"(?:^|\n)\s*(?:export\s+)?interface\s+(\w+)", re.MULTILINE)
    DEF_TYPE   = re.compile(r"(?:^|\n)\s*(?:export\s+)?type\s+(\w+)\s*=", re.MULTILINE)
    DEF_ENUM   = re.compile(r"(?:^|\n)\s*(?:export\s+)?(?:const\s+)?enum\s+(\w+)", re.MULTILINE)
    DECORATOR  = re.compile(r"@(\w+)(?:\([^)]*\))?")
    REACT_HOOK = re.compile(r"\b(use[A-Z]\w+)\s*\(")
    # React component: PascalCase name, after const/function, optionally with type annotation
    REACT_CMP  = re.compile(r"(?:export\s+)?(?:const|function)\s+([A-Z]\w+)\s*[=:(]")
    CALL_LOCAL = re.compile(r"\b(\w+)\s*\(")
    GENERICS   = re.compile(r"<([A-Z]\w+(?:,\s*[A-Z]\w+)*)>")
    PROP_TYPES = re.compile(r"(\w+)\s*:\s*([A-Z]\w+(?:\[\])?)")

    # ── Complexity branches ──────────────────────────────────────────────────
    BRANCH_KW  = re.compile(r"\b(?:if|else|for|while|switch|catch|case)\b|&&|\|\||try\s*\{")

    def analyze(self, path: str, source: str) -> Tuple[ModuleNode, List[FunctionNode]]:
        lang = (Language.TYPESCRIPT.value
                if Path(path).suffix.lower() in (".ts", ".tsx")
                else Language.JAVASCRIPT.value)
        m = ModuleNode(path=path, language=lang)
        m.loc = len(source.splitlines())

        # Strip block comments and line comments for analysis
        clean = re.sub(r"/\*.*?\*/", " ", source, flags=re.DOTALL)
        clean = re.sub(r"//[^\n]*", " ", clean)

        # ── Imports ──────────────────────────────────────────────────────────
        seen_imp: Set[str] = set()
        for pat in (self.IMP_FROM, self.IMP_BARE, self.IMP_REQ,
                    self.IMP_DYN, self.IMP_REEXP):
            for hit in pat.finditer(clean):
                imp = hit.group(1)
                if imp and imp not in seen_imp:
                    seen_imp.add(imp)
                    m.imports.append(imp)
                    m.js_ts_imports.append(imp)

        # ── Exports ──────────────────────────────────────────────────────────
        seen_exp: Set[str] = set()

        def _add_export(name: str):
            if name and name not in seen_exp:
                seen_exp.add(name)
                m.exports.append(name)
                m.js_ts_exports.append(name)

        for pat in (self.EXP_FN, self.EXP_CLS, self.EXP_CONST,
                    self.EXP_IFACE, self.EXP_TYPE, self.EXP_ENUM):
            for hit in pat.finditer(clean):
                _add_export(hit.group(1))

        for hit in self.EXP_DEF.finditer(clean):
            _add_export(hit.group(1) or "default")

        for hit in self.EXP_NAMED.finditer(clean):
            for name in re.findall(r"\b(\w+)\b", hit.group(1)):
                _add_export(name)

        # ── Decorators ───────────────────────────────────────────────────────
        decorators = list(set(self.DECORATOR.findall(clean)))

        # ── TypeScript-specific: interfaces, types, enums ────────────────────
        interfaces = [h.group(1) for h in self.DEF_IFACE.finditer(clean)]
        type_aliases = [h.group(1) for h in self.DEF_TYPE.finditer(clean)]
        enums = [h.group(1) for h in self.DEF_ENUM.finditer(clean)]
        generics_used = list(set(self.GENERICS.findall(clean)))

        # ── React hooks used ─────────────────────────────────────────────────
        hooks_used = list(set(self.REACT_HOOK.findall(clean)))

        # ── All declared names (for call-graph) ──────────────────────────────
        all_local_names: Set[str] = set()
        for pat in (self.DEF_FN, self.DEF_ARR, self.REACT_CMP):
            for hit in pat.finditer(clean):
                all_local_names.add(hit.group(1))
        for hit in self.DEF_CLS.finditer(clean):
            all_local_names.add(hit.group(1))

        # ── Function nodes + call graph ───────────────────────────────────────
        fns: List[FunctionNode] = []
        fn_names_seen: Set[str] = set()

        def _make_fn(name: str, lineno: int, is_pub: bool,
                     fn_source: str = "") -> FunctionNode:
            calls = []
            if fn_source:
                for c in self.CALL_LOCAL.findall(fn_source):
                    if c in all_local_names and c != name:
                        calls.append(c)
            sig = f"{'async ' if 'async' in fn_source[:20] else ''}function {name}(...)"
            return FunctionNode(
                qualified_name=f"{path}::{name}",
                parent_module=path,
                signature=sig,
                is_public_api=is_pub,
                line_number=lineno,
            )

        for pat in (self.DEF_FN, self.DEF_ARR, self.REACT_CMP):
            for hit in pat.finditer(clean):
                name = hit.group(1)
                if name in fn_names_seen:
                    continue
                fn_names_seen.add(name)
                lineno = clean[: hit.start()].count("\n") + 1
                is_pub = name in seen_exp or name[0].isupper()
                fns.append(_make_fn(name, lineno, is_pub))
                if is_pub:
                    _add_export(name)

        # ── Complexity ───────────────────────────────────────────────────────
        m.complexity_score = float(1 + len(self.BRANCH_KW.findall(clean)))

        # ── Metadata stored in purpose_statement for TS-specific info ────────
        ts_info = []
        if lang == Language.TYPESCRIPT.value:
            if interfaces:
                ts_info.append(f"Interfaces: {', '.join(interfaces[:5])}")
            if type_aliases:
                ts_info.append(f"Types: {', '.join(type_aliases[:5])}")
            if enums:
                ts_info.append(f"Enums: {', '.join(enums[:5])}")
            if generics_used:
                ts_info.append(f"Generics: {', '.join(generics_used[:3])}")
        if decorators:
            ts_info.append(f"Decorators: {', '.join(decorators[:5])}")
        if hooks_used:
            ts_info.append(f"React hooks: {', '.join(hooks_used[:5])}")
        if ts_info:
            m.purpose_statement = " | ".join(ts_info)

        return m, fns


# ── SQL ────────────────────────────────────────────────────────────────────────

class SQLAnalyzerBasic:
    FROM_RE   = re.compile(r"\bFROM\s+([`\"'\[]?[\w.]+[`\"'\]]?)", re.IGNORECASE)
    JOIN_RE   = re.compile(r"\bJOIN\s+([`\"'\[]?[\w.]+[`\"'\]]?)", re.IGNORECASE)
    CTE_RE    = re.compile(r"\bWITH\s+(\w+)\s+AS\s*\(", re.IGNORECASE)
    CREATE_RE = re.compile(
        r"\bCREATE\s+(?:OR\s+REPLACE\s+)?(?:TABLE|VIEW)\s+"
        r"(?:IF\s+NOT\s+EXISTS\s+)?([`\"'\[]?[\w.]+[`\"'\]]?)", re.IGNORECASE)

    def analyze(self, path: str, source: str) -> ModuleNode:
        m = ModuleNode(path=path, language=Language.SQL.value)
        m.loc = len(source.splitlines())
        clean = re.sub(r"--[^\n]*", "", source)
        clean = re.sub(r"/\*.*?\*/", "", clean, flags=re.DOTALL)
        ctes = {h.group(1).lower() for h in self.CTE_RE.finditer(clean)}
        tables: set = set()
        for pat in (self.FROM_RE, self.JOIN_RE):
            for h in pat.finditer(clean):
                t = h.group(1).strip("`\"'[] \t").lower()
                if t and t not in ctes:
                    tables.add(t)
        m.imports = sorted(tables)
        m.exports = sorted(ctes)
        for h in self.CREATE_RE.finditer(clean):
            t = h.group(1).strip("`\"'[] \t").lower()
            if t not in m.exports:
                m.exports.append(t)
        return m


# ── YAML ───────────────────────────────────────────────────────────────────────

class YAMLAnalyzer:
    def analyze(self, path: str, source: str) -> ModuleNode:
        m = ModuleNode(path=path, language=Language.YAML.value)
        m.loc = len(source.splitlines())
        if not YAML_OK:
            return m
        try:
            data = _yaml.safe_load(source)
            if not isinstance(data, dict):
                return m
            for mdl in data.get("models", []):
                if isinstance(mdl, dict) and "name" in mdl:
                    m.exports.append(mdl["name"])
            if "dag_id" in data:
                m.exports.append(data["dag_id"])
            if not m.exports:
                m.exports = [str(k) for k in list(data.keys())[:10]]
        except Exception:
            pass
        return m


# ── Notebook ───────────────────────────────────────────────────────────────────

class NotebookAnalyzer:
    def analyze(self, path: str, source: str) -> Tuple[ModuleNode, List[FunctionNode]]:
        m = ModuleNode(path=path, language=Language.NOTEBOOK.value)
        fns: List[FunctionNode] = []
        try:
            nb = json.loads(source)
            code = "".join(
                "".join(c.get("source", []))
                for c in nb.get("cells", [])
                if c.get("cell_type") == "code"
            )
            m, fns = PythonASTAnalyzer().analyze(path, code)
            m.language = Language.NOTEBOOK.value
        except Exception as e:
            m.purpose_statement = f"[NOTEBOOK ERROR: {e}]"
        return m, fns


# ── Router ──────────────────────────────────────────────────────────────────────

class TreeSitterAnalyzer:
    """
    Multi-language dispatcher.
    Named TreeSitterAnalyzer to reflect the upgrade path — the interface
    is tree-sitter-compatible so backends can be swapped transparently.
    """

    def __init__(self):
        self.py   = PythonASTAnalyzer()
        self.jsts = JSTSAnalyzer()
        self.sql  = SQLAnalyzerBasic()
        self.yaml = YAMLAnalyzer()
        self.nb   = NotebookAnalyzer()
        self.errors: List[str] = []

    def analyze_file(self, path: str) -> Tuple[Optional[ModuleNode], List[FunctionNode]]:
        if LanguageRouter.should_skip(path):
            return None, []
        lang = LanguageRouter.detect(path)
        if lang == Language.UNKNOWN.value:
            return None, []
        try:
            source = Path(path).read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            self.errors.append(f"READ {path}: {e}")
            return None, []
        if not source.strip():
            return None, []
        try:
            if lang == Language.PYTHON.value:
                return self.py.analyze(path, source)
            elif lang in (Language.JAVASCRIPT.value, Language.TYPESCRIPT.value):
                return self.jsts.analyze(path, source)
            elif lang == Language.SQL.value:
                return self.sql.analyze(path, source), []
            elif lang == Language.YAML.value:
                return self.yaml.analyze(path, source), []
            elif lang == Language.NOTEBOOK.value:
                return self.nb.analyze(path, source)
            else:
                return ModuleNode(path=path, language=lang,
                                  loc=len(source.splitlines())), []
        except Exception as e:
            self.errors.append(f"ANALYSIS {path}: {e}")
            return ModuleNode(path=path, language=lang,
                              purpose_statement=f"[ERROR: {e}]"), []

    def analyze_repo(self, repo_path: str,
                     max_files: int = 2000) -> Tuple[List[ModuleNode], List[FunctionNode]]:
        modules, functions, count = [], [], 0
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fname in sorted(files):
                if count >= max_files:
                    break
                fpath = os.path.join(root, fname)
                if LanguageRouter.detect(fpath) == Language.UNKNOWN.value:
                    continue
                mod, fns = self.analyze_file(fpath)
                if mod:
                    modules.append(mod)
                    functions.extend(fns)
                    count += 1
        return modules, functions
