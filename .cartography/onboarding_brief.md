# FDE Day-One Onboarding Brief
## Codebase: cartographer_8faztmtx
*Generated: 2026-03-11 18:11 UTC by Brownfield Cartographer*

---

## Executive Summary

This brief covers `cartographer_8faztmtx`, a codebase with 8 modules
and 8 tracked datasets.

---

## The Five FDE Day-One Questions

### 🔵 1. What is the primary data ingestion path?

Entry points (in-degree=0 datasets): customer_orders, customer_payments, final, orders, payments

### 🟢 2. What are the 3-5 most critical output datasets?

Output sinks (out-degree=0 datasets): Not detected

### 🔴 3. What is the blast radius if the critical module fails?

Highest-impact modules by PageRank: orders.sql, dbt_project.yml, customers.sql. Run blast_radius() for full impact.

### 🟡 4. Where is business logic concentrated vs. distributed?

Logic concentrated in: orders.sql, dbt_project.yml, customers.sql. Circular deps: None.

### ⚡ 5. What has changed most frequently (git velocity)?

Highest-velocity files (30d): No git history. Active pain points.

---

## Architecture Snapshot

| Metric | Value |
|--------|-------|
| Total Modules | 8 |
| Total Datasets | 8 |
| Transformations | 5 |
| Circular Deps | 0 |
| Doc Drift Flags | 0 |

## Top 5 Architectural Hubs (by PageRank)

1. `C:\Users\reus\AppData\Local\Temp\cartographer_8faztmtx\models\orders.sql` — PageRank: 0.20904 | Domain: other
2. `C:\Users\reus\AppData\Local\Temp\cartographer_8faztmtx\dbt_project.yml` — PageRank: 0.11299 | Domain: other
3. `C:\Users\reus\AppData\Local\Temp\cartographer_8faztmtx\models\customers.sql` — PageRank: 0.11299 | Domain: other
4. `C:\Users\reus\AppData\Local\Temp\cartographer_8faztmtx\models\schema.yml` — PageRank: 0.11299 | Domain: other
5. `C:\Users\reus\AppData\Local\Temp\cartographer_8faztmtx\models\staging\schema.yml` — PageRank: 0.11299 | Domain: other

## Data Lineage Summary

**Source Tables / Files:** customer_orders, customer_payments, final, orders, payments, order_payments, renamed, the
**Output Tables / Files:** None detected

---

*This brief was auto-generated. Verify critical path answers against live codebase.*