#!/usr/bin/env python3
"""
Brownfield Cartographer CLI
Usage:
  cartographer analyze <repo_path_or_url> [--output-dir <dir>] [--api-key <key>] [--incremental] [--dashboard]
  cartographer query <repo_path> [--interactive]
  cartographer blast-radius <repo_path> <module_path>
  cartographer lineage <repo_path> <dataset_name> [--direction upstream|downstream]
  cartographer dashboard <repo_path> [--port 7842]
"""
from __future__ import annotations
import argparse
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def cmd_analyze(args):
    from src.orchestrator import Orchestrator

    repo_path = args.repo
    if repo_path.startswith('http') or repo_path.startswith('git@'):
        orch = Orchestrator(".", output_dir=args.output_dir, api_key=args.api_key or "")
        repo_path = orch.clone_if_url(repo_path)

    orch = Orchestrator(
        repo_path=repo_path,
        output_dir=args.output_dir,
        api_key=args.api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
    )
    result = orch.run(incremental=args.incremental)

    if getattr(args, 'dashboard', False):
        _launch_dashboard(orch, result, args)

    return result, orch


def cmd_dashboard(args):
    """Run analysis (or reuse existing) then launch the dashboard."""
    from src.orchestrator import Orchestrator

    repo_path = args.repo
    if repo_path.startswith('http') or repo_path.startswith('git@'):
        orch = Orchestrator(".", output_dir=args.output_dir, api_key=args.api_key or "")
        repo_path = orch.clone_if_url(repo_path)

    orch = Orchestrator(
        repo_path=repo_path,
        output_dir=args.output_dir,
        api_key=args.api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
    )

    # Reuse existing analysis if available and not forced
    if not args.reanalyze and orch._has_previous_run():
        print(f"  [Dashboard] Reusing existing analysis from {orch.output_dir}")
        result = orch.run()  # fast re-run to rebuild KG in memory
    else:
        result = orch.run()

    _launch_dashboard(orch, result, args)


def _launch_dashboard(orch, result, args):
    """Start the Flask dashboard server."""
    from src.dashboard.server import serve
    port = getattr(args, 'port', 7842)
    host = getattr(args, 'host', '127.0.0.1')
    no_browser = getattr(args, 'no_browser', False)
    serve(
        kg=orch.kg,
        result=result,
        api_key=orch.api_key,
        host=host,
        port=port,
        open_browser=not no_browser,
    )


def cmd_query(args):
    """Interactive query mode."""
    from src.orchestrator import Orchestrator
    import json as _json

    output_dir = args.output_dir or os.path.join(args.repo, ".cartography")
    result_path = os.path.join(output_dir, "cartography_result.json")

    if not os.path.exists(result_path):
        print(f"No analysis found at {output_dir}. Run 'analyze' first.")
        sys.exit(1)

    # Re-run analysis to rebuild KG (quick since files already exist)
    orch = Orchestrator(
        repo_path=args.repo,
        output_dir=output_dir,
        api_key=args.api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
    )
    result = orch.run()
    nav = orch.get_navigator()

    if args.interactive:
        print("\n🔍 Navigator — Interactive Query Mode")
        print("Commands: find <concept> | lineage <dataset> | blast <module> | explain <module> | quit")
        print()
        while True:
            try:
                query = input("Navigator> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query or query in ('quit', 'exit', 'q'):
                break
            if query.startswith('find '):
                r = nav.find_implementation(query[5:])
                print(json.dumps(r, indent=2, default=str))
            elif query.startswith('lineage '):
                parts = query[8:].split()
                dataset = parts[0]
                direction = parts[1] if len(parts) > 1 else "upstream"
                r = nav.trace_lineage(dataset, direction)
                print(json.dumps(r, indent=2, default=str))
            elif query.startswith('blast '):
                r = nav.blast_radius(query[6:])
                print(json.dumps(r, indent=2, default=str))
            elif query.startswith('explain '):
                r = nav.explain_module(query[8:])
                print(json.dumps(r, indent=2, default=str))
            else:
                r = nav.interactive_query(query)
                print(r)
    else:
        if args.query:
            r = nav.interactive_query(args.query)
            print(r)


def main():
    parser = argparse.ArgumentParser(
        description="Brownfield Cartographer — Codebase Intelligence System"
    )
    subparsers = parser.add_subparsers(dest='command')

    # analyze subcommand
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a codebase')
    analyze_parser.add_argument('repo', help='Local path or GitHub URL')
    analyze_parser.add_argument('--output-dir', '-o', default=None, help='Output directory for artifacts')
    analyze_parser.add_argument('--api-key', '-k', default=None, help='Anthropic API key')
    analyze_parser.add_argument('--incremental', action='store_true', help='Only re-analyze changed files')
    analyze_parser.add_argument('--dashboard', action='store_true', help='Launch dashboard after analysis')
    analyze_parser.add_argument('--port', type=int, default=7842, help='Dashboard port (default: 7842)')
    analyze_parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')

    # dashboard subcommand
    dash_parser = subparsers.add_parser('dashboard', help='Launch interactive dashboard (analyze if needed)')
    dash_parser.add_argument('repo', help='Local path or GitHub URL')
    dash_parser.add_argument('--output-dir', '-o', default=None)
    dash_parser.add_argument('--api-key', '-k', default=None)
    dash_parser.add_argument('--port', type=int, default=7842, help='Port to serve on (default: 7842)')
    dash_parser.add_argument('--host', default='127.0.0.1', help='Host to bind (default: 127.0.0.1)')
    dash_parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    dash_parser.add_argument('--reanalyze', action='store_true', help='Force re-analysis even if cache exists')

    # query subcommand
    query_parser = subparsers.add_parser('query', help='Query analyzed codebase')
    query_parser.add_argument('repo', help='Path to analyzed repo')
    query_parser.add_argument('--output-dir', '-o', default=None)
    query_parser.add_argument('--api-key', '-k', default=None)
    query_parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    query_parser.add_argument('--query', '-q', default=None, help='Single query string')

    # blast-radius subcommand
    blast_parser = subparsers.add_parser('blast-radius', help='Show blast radius for a module')
    blast_parser.add_argument('repo', help='Path to analyzed repo')
    blast_parser.add_argument('module', help='Module path')
    blast_parser.add_argument('--output-dir', '-o', default=None)

    # lineage subcommand
    lineage_parser = subparsers.add_parser('lineage', help='Trace dataset lineage')
    lineage_parser.add_argument('repo', help='Path to analyzed repo')
    lineage_parser.add_argument('dataset', help='Dataset name')
    lineage_parser.add_argument('--direction', default='upstream', choices=['upstream', 'downstream'])
    lineage_parser.add_argument('--output-dir', '-o', default=None)

    args = parser.parse_args()

    if args.command == 'analyze':
        cmd_analyze(args)

    elif args.command == 'dashboard':
        cmd_dashboard(args)

    elif args.command == 'query':
        cmd_query(args)

    elif args.command == 'blast-radius':
        from src.orchestrator import Orchestrator
        orch = Orchestrator(args.repo, output_dir=args.output_dir)
        orch.run()
        nav = orch.get_navigator()
        result = nav.blast_radius(args.module)
        print(json.dumps(result, indent=2, default=str))

    elif args.command == 'lineage':
        from src.orchestrator import Orchestrator
        orch = Orchestrator(args.repo, output_dir=args.output_dir)
        orch.run()
        nav = orch.get_navigator()
        result = nav.trace_lineage(args.dataset, args.direction)
        print(json.dumps(result, indent=2, default=str))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
