#!/usr/bin/env python3
"""
G3: Deep Research Agent with Memory Constraints
───────────────────────────────────────────────
Entry point — supports CLI and interactive modes.

Usage:
    python main.py                                 # interactive
    python main.py --mode research -q "query"      # one-shot
    python main.py --mode sessions                 # list history
"""

import argparse
import os
import sys

import yaml

from src.pipeline.agent import ResearchAgent


def load_config(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        print("Copy config.example.yaml → config.yaml and add your Dify API keys.")
        sys.exit(1)
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def interactive(agent: ResearchAgent):
    print()
    print("=" * 64)
    print("  G3: Deep Research Agent")
    print("  Hierarchical Memory · Token-Budgeted · Cost-Capped")
    print("=" * 64)
    print()
    print("Commands:")
    print("  research <query>   — Start a new research session")
    print("  sessions           — List past sessions")
    print("  view <session_id>  — View a past session in detail")
    print("  quit               — Exit")
    print()

    while True:
        try:
            raw = input("→ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw:
            continue

        low = raw.lower()
        if low in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        elif low == "sessions":
            agent.list_sessions()
        elif low.startswith("view "):
            agent.view_session(raw[5:].strip())
        elif low.startswith("research "):
            query = raw[9:].strip()
            if query:
                agent.research(query)
            else:
                print("  Please provide a query.")
        else:
            # Treat bare text as a research query
            agent.research(raw)


def main():
    ap = argparse.ArgumentParser(
        description="G3: Deep Research Agent with Memory Constraints"
    )
    ap.add_argument(
        "--mode",
        choices=["research", "interactive", "sessions"],
        default="interactive",
    )
    ap.add_argument("--query", "-q", type=str, help="Research query")
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    config = load_config(args.config)
    agent = ResearchAgent(config)

    try:
        if args.mode == "research":
            if not args.query:
                print("Error: --query is required for research mode.")
                sys.exit(1)
            result = agent.research(args.query, verbose=not args.quiet)
            if args.quiet:
                print(result["answer"])
        elif args.mode == "sessions":
            agent.list_sessions()
        else:
            interactive(agent)
    finally:
        agent.close()


if __name__ == "__main__":
    main()