"""
SQL lineage: table-level AND column-level analysis.
  Table-level:  regex always + sqlglot when installed
  Column-level: SELECT clause parsing -> {target_col: [source_cols]}
Dialects supported: PostgreSQL, BigQuery, Snowflake, DuckDB, dbt (Jinja-aware).
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models import TransformationNode, TransformationType

try:
    import sqlglot
    import sqlglot.expressions as exp
    SQLGLOT_OK = True
except ImportError:
    SQLGLOT_OK = False

# ── Shared regex ──────────────────────────────────────────────────────────────
FROM_RE   = re.compile(r"\bFROM\s+([`\"'\[]?[\w.]+[`\"'\]]?)",   re.IGNORECASE)
JOIN_RE   = re.compile(r"\bJOIN\s+([`\"'\[]?[\w.]+[`\"'\]]?)",   re.IGNORECASE)
INTO_RE   = re.compile(r"\bINTO\s+([`\"'\[]?[\w.]+[`\"'\]]?)",   re.IGNORECASE)
CTE_RE    = re.compile(r"\bWITH\s+(\w+)\s+AS\s*\(",              re.IGNORECASE)
MULTI_CTE_RE = re.compile(r",\s*(\w+)\s+AS\s*\(",                   re.IGNORECASE)
CREATE_RE = re.compile(
    r"\bCREATE\s+(?:OR\s+REPLACE\s+)?(?:TABLE|VIEW)\s+"
    r"(?:IF\s+NOT\s+EXISTS\s+)?([`\"'\[]?[\w.]+[`\"'\]]?)",      re.IGNORECASE)
INSERT_RE = re.compile(r"\bINSERT\s+(?:INTO\s+)?([`\"'\[]?[\w.]+[`\"'\]]?)", re.IGNORECASE)
SELECT_BLOCK_RE = re.compile(r"\bSELECT\b(.*?)\bFROM\b", re.IGNORECASE | re.DOTALL)

SQL_KEYWORDS = {
    "select","from","where","and","or","not","in","is","null","case",
    "when","then","else","end","over","partition","by","order","asc",
    "desc","distinct","count","sum","avg","max","min","coalesce",
    "ifnull","isnull","cast","convert","date","timestamp","true","false",
}



# Single-word SQL fragments that can leak as fake table names after Jinja strip
_JUNK_SOURCES = {
    "placeholder_tbl", "the", "final", "renamed", "staged", "base",
    "unioned", "joined", "filtered", "pivoted", "deduplicated",
    "a", "b", "c", "t", "s", "l", "r",
}

def _is_real_source(name: str) -> bool:
    """Return True if this looks like a real table/dataset name, not a CTE fragment."""
    n = name.lower().strip()
    if n in _JUNK_SOURCES:
        return False
    if len(n) <= 2:
        return False
    # pure SQL keyword
    if n in SQL_KEYWORDS:
        return False
    return True

def _strip(raw: str) -> str:
    return raw.strip("`\"'[] \t\n").lower()


def _clean(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return sql


# ── Column-level lineage ──────────────────────────────────────────────────────

def _split_cols(clause: str) -> List[str]:
    """Split SELECT clause on commas, respecting parentheses."""
    cols, depth, buf = [], 0, []
    for ch in clause:
        if ch == "(":   depth += 1; buf.append(ch)
        elif ch == ")": depth -= 1; buf.append(ch)
        elif ch == "," and depth == 0:
            cols.append("".join(buf).strip()); buf = []
        else:
            buf.append(ch)
    if buf:
        cols.append("".join(buf).strip())
    return cols


def _parse_col(expr: str) -> Tuple[str, List[str]]:
    """Parse single SELECT column expression -> (target_alias, [source_cols])."""
    expr = expr.strip()
    # table.col AS alias
    m = re.match(r"(\w+)\.(\w+)\s+AS\s+(\w+)$", expr, re.IGNORECASE)
    if m:
        return m.group(3).lower(), [m.group(2).lower()]
    # any_expr AS alias
    m = re.match(r"(.+)\s+AS\s+(\w+)$", expr, re.IGNORECASE)
    if m:
        target = m.group(2).lower()
        raw_cols = re.findall(r"(?:\w+\.)?(\w+)", m.group(1))
        cols = [c.lower() for c in raw_cols
                if c.lower() not in SQL_KEYWORDS and not c.isdigit()]
        return target, cols or ["<expr>"]
    # table.col
    m = re.match(r"(\w+)\.(\w+)$", expr)
    if m:
        return m.group(2).lower(), [m.group(2).lower()]
    # bare col
    m = re.match(r"^(\w+)$", expr)
    if m and m.group(1).lower() not in SQL_KEYWORDS:
        return m.group(1).lower(), [m.group(1).lower()]
    # complex expr — extract any column refs
    raw_cols = re.findall(r"(?:\w+\.)?(\w+)", expr)
    cols = [c.lower() for c in raw_cols
            if c.lower() not in SQL_KEYWORDS and not c.isdigit()]
    return "<complex>", cols or ["<expr>"]


def extract_column_mappings(sql: str) -> Dict[str, List[str]]:
    """
    Parse SELECT clause(s) -> {target_column: [source_columns]}.
    Best-effort: complex expressions recorded as ["<expr>"].
    """
    mappings: Dict[str, List[str]] = {}
    for sel in SELECT_BLOCK_RE.finditer(_clean(sql)):
        for col_expr in _split_cols(sel.group(1)):
            col_expr = col_expr.strip()
            if not col_expr or col_expr == "*":
                continue
            target, sources = _parse_col(col_expr)
            if target and target != "<complex>":
                mappings[target] = sources
    return mappings


# ── Table-level lineage ───────────────────────────────────────────────────────

class SQLLineageAnalyzer:
    """
    Extracts table-level AND column-level lineage from SQL.
    Uses sqlglot when available, regex otherwise.
    """
    DIALECTS = ["duckdb", "bigquery", "snowflake", "postgres", None]

    def extract_lineage(self, sql: str, source_file: str = "",
                        target_name: str = "") -> List[TransformationNode]:
        if SQLGLOT_OK:
            result = self._sqlglot(sql, source_file, target_name)
            if result:
                return result
        return self._regex(sql, source_file, target_name)

    # ── sqlglot path ──────────────────────────────────────────────────────────
    def _sqlglot(self, sql: str, source_file: str,
                 target_name: str) -> List[TransformationNode]:
        parsed = None
        for dialect in self.DIALECTS:
            try:
                parsed = sqlglot.parse(
                    sql, dialect=dialect,
                    error_level=sqlglot.ErrorLevel.IGNORE)
                if parsed:
                    break
            except Exception:
                continue
        if not parsed:
            return []

        results = []
        for i, stmt in enumerate(parsed):
            if stmt is None:
                continue
            sources: Set[str] = set()
            targets: Set[str] = set()
            ctes:    Set[str] = set()
            for cte in stmt.find_all(exp.CTE):
                if hasattr(cte, "alias"):
                    ctes.add(cte.alias.lower())
            for tbl in stmt.find_all(exp.Table):
                name = (tbl.name or "").lower()
                if name and name not in ctes:
                    sources.add(name)
            for into in stmt.find_all(exp.Into):
                if hasattr(into, "this") and hasattr(into.this, "name"):
                    t = into.this.name.lower()
                    if t:
                        targets.add(t); sources.discard(t)
            for create in stmt.find_all(exp.Create):
                if hasattr(create, "this") and hasattr(create.this, "name"):
                    t = create.this.name.lower()
                    if t:
                        targets.add(t); sources.discard(t)
            if target_name and not targets:
                targets.add(target_name)
            if sources or targets:
                col_maps = extract_column_mappings(sql)
                name = target_name or f"sql_{Path(source_file).stem}_{i}"
                results.append(TransformationNode(
                    name=name,
                    source_datasets=sorted(sources - targets),
                    target_datasets=sorted(targets),
                    transformation_type=TransformationType.SQL_SELECT.value,
                    source_file=source_file,
                    sql_query_if_applicable=sql[:500],
                    column_mappings=col_maps,
                ))
        return results

    # ── regex path ────────────────────────────────────────────────────────────
    def _regex(self, sql: str, source_file: str,
               target_name: str) -> List[TransformationNode]:
        clean = _clean(sql)
        ctes: Set[str]    = {m.group(1).lower() for m in CTE_RE.finditer(clean)}
        ctes |= {m.group(1).lower() for m in MULTI_CTE_RE.finditer(clean)}
        sources: Set[str] = set()
        targets: Set[str] = set()
        for pat in (FROM_RE, JOIN_RE):
            for m in pat.finditer(clean):
                t = _strip(m.group(1))
                if t and t not in ctes and _is_real_source(t):
                    sources.add(t)
        for pat in (INTO_RE, INSERT_RE, CREATE_RE):
            for m in pat.finditer(clean):
                t = _strip(m.group(1))
                if t and len(t) > 1:
                    targets.add(t)
        if target_name:
            targets.add(target_name)
        sources -= targets
        if not sources and not targets:
            return []
        col_maps = extract_column_mappings(sql)
        name = target_name or f"sql_{Path(source_file).stem}"
        return [TransformationNode(
            name=name,
            source_datasets=sorted(sources),
            target_datasets=sorted(targets),
            transformation_type=TransformationType.SQL_SELECT.value,
            source_file=source_file,
            sql_query_if_applicable=sql[:500],
            column_mappings=col_maps,
        )]

    # ── dbt model ─────────────────────────────────────────────────────────────
    def analyze_dbt_model(self, path: str) -> List[TransformationNode]:
        try:
            raw = Path(path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []
        ref_re    = re.compile(r"\{\{\s*ref\(['\"](\w+)['\"]\)\s*\}\}")
        source_re = re.compile(r"\{\{\s*source\(['\"](\w+)['\"],\s*['\"](\w+)['\"]\)\s*\}\}")
        dbt_sources = [m.group(1).lower() for m in ref_re.finditer(raw)]
        dbt_sources += [f"{m.group(1)}.{m.group(2)}".lower()
                        for m in source_re.finditer(raw)]
        # Strip Jinja then parse SQL
        clean_sql = re.sub(r"\{\{[^}]+\}\}", "placeholder_tbl", raw)
        clean_sql = re.sub(r"\{%[^%]+%\}", "", clean_sql)
        target = Path(path).stem.lower()
        # Extract ALL CTE names from cleaned SQL — these are never real source tables
        all_ctes: Set[str] = {m.group(1).lower() for m in CTE_RE.finditer(clean_sql)}
        all_ctes |= {m.group(1).lower() for m in MULTI_CTE_RE.finditer(clean_sql)}
        all_ctes.add("placeholder_tbl")
        transforms = self.extract_lineage(clean_sql, path, target_name=target)
        # Only trust dbt ref()/source() — they are the real dependencies
        # Supplement with regex finds but strip CTEs and junk
        regex_sources: Set[str] = set()
        for t in transforms:
            regex_sources.update(t.source_datasets)
        regex_sources -= all_ctes
        regex_sources = {s for s in regex_sources if _is_real_source(s)}
        # dbt ref() sources are authoritative; use regex as fallback only if no refs found
        all_sources: Set[str] = set(dbt_sources) if dbt_sources else regex_sources
        if dbt_sources:
            # Still add regex sources that look like real table refs (not CTEs)
            all_sources |= {s for s in regex_sources if s not in all_ctes and "_" in s}
        all_sources.discard(target)
        all_sources -= all_ctes
        col_maps = extract_column_mappings(clean_sql)
        return [TransformationNode(
            name=f"dbt_model_{target}",
            source_datasets=sorted(all_sources),
            target_datasets=[target],
            transformation_type=TransformationType.DBT_MODEL.value,
            source_file=path,
            sql_query_if_applicable=raw[:500],
            column_mappings=col_maps,
        )]

    def analyze_file(self, path: str) -> List[TransformationNode]:
        try:
            sql = Path(path).read_text(encoding="utf-8", errors="replace")
            return self.extract_lineage(sql, path)
        except Exception:
            return []