import uuid

from ..memory.store import MemoryStore
from ..memory.manager import MemoryManager
from ..dify_client import DifyClient
from ..research.decomposer import QueryDecomposer
from ..research.executor import ResearchExecutor
from ..research.synthesizer import Synthesizer


class ResearchAgent:
    """
    Orchestrates the full research pipeline:

        decompose → (research + memorise) × N → synthesize

    All LLM interactions are routed through Dify workflows.
    Memory constraints are enforced by MemoryManager at every step.
    """

    def __init__(self, config: dict):
        self.config = config
        dify = config["dify"]
        max_sq = config.get("constraints", {}).get("max_sub_questions", 5)

        self.store = MemoryStore()
        self.memory = MemoryManager(config, self.store)

        self.decomposer = QueryDecomposer(
            DifyClient(dify["base_url"], dify["decomposer_api_key"]),
            max_sub_questions=max_sq,
        )
        self.executor = ResearchExecutor(
            DifyClient(dify["base_url"], dify["researcher_api_key"]),
            self.memory,
        )
        self.synthesizer = Synthesizer(
            DifyClient(dify["base_url"], dify["synthesizer_api_key"]),
            self.memory,
        )

    # ── Main entry point ────────────────────────────────────────────

    def research(self, query: str, verbose: bool = True) -> dict:
        session_id = uuid.uuid4().hex[:8]
        self.store.create_session(session_id, query)

        if verbose:
            self._header(session_id, query)

        # ── Stage 1: decompose ──
        if verbose:
            print("📋 Stage 1: Decomposing query…")

        decomp = self.decomposer.decompose(query)
        sub_qs = decomp["sub_questions"]
        self.store.store_sub_questions(session_id, sub_qs)
        self.memory.record_cost(
            session_id,
            decomp.get("input_tokens", 0),
            decomp.get("output_tokens", 0),
        )

        if verbose:
            print(f"  Strategy: {decomp['strategy']}")
            print(f"  Sub-questions ({len(sub_qs)}):")
            for i, q in enumerate(sub_qs, 1):
                print(f"    {i}. {q}")
            print()

        # ── Stage 2: research each sub-question ──
        if verbose:
            print("🔍 Stage 2: Researching sub-questions…\n")

        for i, sq in enumerate(sub_qs):
            remaining, within = self.memory.check_budget(session_id)
            if not within:
                if verbose:
                    print(
                        f"  ⛔ Budget exhausted (${self.memory.max_cost_usd} limit). "
                        f"Stopping at question {i + 1}/{len(sub_qs)}."
                    )
                for j in range(i, len(sub_qs)):
                    self.store.update_sub_question_status(
                        session_id, j, "skipped_budget"
                    )
                break

            if verbose:
                print(f"  [{i + 1}/{len(sub_qs)}] {sq[:80]}…")

            result = self.executor.research(session_id, sq, query)
            status = (
                "completed"
                if not result.get("budget_exceeded")
                else "skipped_budget"
            )
            self.store.update_sub_question_status(session_id, i, status)

            if verbose:
                stats = self.memory.get_memory_stats(session_id)
                finding_preview = result["finding"][:120].replace("\n", " ")
                print(f"       Finding: {finding_preview}…")
                print(
                    f"       Memory: {stats['episodic_entries']}/"
                    f"{stats['episodic_capacity']} episodic, "
                    f"{stats['compressed_entries']} compressed, "
                    f"{stats['total_memory_tokens']} tok stored"
                )
                print(
                    f"       Context used: {result['memory_context_tokens']}/"
                    f"{self.memory.memory_budget} tok budget"
                )
                print(f"       Cost remaining: ${stats['cost_remaining_usd']:.4f}")
                for c in result["memory_update"].get("compressed", []):
                    print(
                        f"       ♻ Compressed #{c['source_id']}: "
                        f"{c['original_tokens']}→{c['compressed_tokens']} tok"
                    )
                print()

        # ── Stage 3: synthesize ──
        if verbose:
            print("🧠 Stage 3: Synthesizing final answer…\n")

        synthesis = self.synthesizer.synthesize(session_id, query)
        self.store.complete_session(session_id, synthesis["answer"])

        final_stats = self.memory.get_memory_stats(session_id)
        session_info = self.store.get_session_summary(session_id)

        if verbose:
            self._report(
                session_id, query, synthesis["answer"],
                final_stats, session_info, synthesis,
            )

        return {
            "session_id": session_id,
            "query": query,
            "answer": synthesis["answer"],
            "sub_questions": sub_qs,
            "stats": final_stats,
            "session": session_info,
        }

    # ── Session browsing ────────────────────────────────────────────

    def list_sessions(self):
        sessions = self.store.get_all_sessions()
        if not sessions:
            print("  No sessions found.")
            return
        print(f"\n{'='*64}")
        print("  RESEARCH SESSIONS")
        print(f"{'='*64}\n")
        for s in sessions:
            icon = "✅" if s["status"] == "completed" else "⏳"
            print(
                f"  {icon} {s['session_id']}  "
                f"${s['estimated_cost_usd']:.4f}  "
                f"{s['started_at'][:19]}"
            )
            print(f"     {s['query'][:80]}")
            print()

    def view_session(self, session_id: str):
        s = self.store.get_session_summary(session_id)
        if not s:
            print(f"  Session {session_id} not found.")
            return
        print(f"\n{'='*64}")
        print(f"  SESSION: {session_id}")
        print(f"{'='*64}\n")
        print(f"  Query:       {s['query']}")
        print(f"  Status:      {s['status']}")
        print(f"  Started:     {s.get('started_at', 'N/A')}")
        print(f"  Completed:   {s.get('completed_at', 'N/A')}")
        print(f"  Cost:        ${s.get('estimated_cost_usd', 0):.6f}")
        print(
            f"  Tokens:      {s.get('total_input_tokens', 0)} in / "
            f"{s.get('total_output_tokens', 0)} out"
        )
        print(f"  Sub-Qs:      {s.get('sub_questions_count', 0)}")
        print(f"  Episodic:    {s.get('episodic_entries', 0)}")
        print(f"  Compressed:  {s.get('compressed_entries', 0)}")
        print()
        if s.get("final_answer"):
            print("📝 Answer:")
            print("─" * 64)
            print(s["final_answer"])
            print("─" * 64)
            print()

    # ── Formatting helpers ──────────────────────────────────────────

    def _header(self, session_id: str, query: str):
        print(f"\n{'='*64}")
        print(f"  RESEARCH SESSION: {session_id}")
        print(f"  Query: {query}")
        print(
            f"  Constraints: ≤{self.memory.max_context_tokens} tok/call, "
            f"≤${self.memory.max_cost_usd}/session"
        )
        print(f"{'='*64}\n")

    def _report(self, session_id, query, answer, stats, session, synthesis):
        print(f"{'='*64}")
        print(f"  RESEARCH COMPLETE — Session {session_id}")
        print(f"{'='*64}\n")

        print("📊 Memory Utilisation:")
        print(
            f"   Episodic buffer:    {stats['episodic_entries']}/"
            f"{stats['episodic_capacity']} entries "
            f"({stats['episodic_tokens']} tok)"
        )
        print(
            f"   Compressed memory:  {stats['compressed_entries']} entries "
            f"({stats['compressed_tokens']} tok)"
        )
        print(f"   Total stored:       {stats['total_memory_tokens']} tok")
        print(f"   Synthesis context:  {synthesis['context_tokens']} tok used")
        print(f"   Per-call budget:    {self.memory.max_context_tokens} tok max")
        print()

        print("💰 Cost:")
        print(f"   Input tokens:  {session.get('total_input_tokens', 0):,}")
        print(f"   Output tokens: {session.get('total_output_tokens', 0):,}")
        print(f"   Total cost:    ${session.get('estimated_cost_usd', 0):.6f}")
        print(f"   Budget limit:  ${self.memory.max_cost_usd}")
        print()

        print("📝 Answer:")
        print("─" * 64)
        print(answer)
        print("─" * 64)
        print()

    def close(self):
        self.store.close()