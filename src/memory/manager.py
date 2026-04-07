import json
import re
from collections import Counter

from ..memory.token_counter import TokenCounter
from ..memory.store import MemoryStore


class MemoryManager:
    """
    Hierarchical memory manager with strict token and cost constraints.

    Memory Levels
    ─────────────
    Level 0  Working Register     Ephemeral (current sub-question context).
                                  Not persisted — exists only as local variables.

    Level 1  Episodic Buffer      Recent raw findings.  Bounded capacity
                                  (default 5 entries × 200 tokens each = 1 000 tokens max).
                                  When capacity is exceeded the oldest entry is
                                  compressed and moved to Level 2.

    Level 2  Compressed Memory    Summarized findings.  Unbounded entry count but
                                  each entry ≤ 100 tokens.  Retrieved via keyword
                                  overlap ranking (top-k).

    Level 3  Session Index        Keyword → memory-ID mapping stored as JSON inside
                                  each entry.  Negligible token overhead.

    Budget Allocation (per LLM call, default 2 000 tokens)
    ──────────────────────────────────────────────────────
    System prompt .............. ~400 tokens
    User query / sub-question .. ~200 tokens
    Memory context ............. ~1 400 tokens  (the remainder)
    (Output tokens are counted for cost but not against the input budget.)
    """

    def __init__(self, config: dict, store: MemoryStore):
        self.config = config
        self.store = store
        self.counter = TokenCounter()

        c = config.get("constraints", {})
        self.max_context_tokens = c.get("max_context_tokens_per_call", 2000)
        self.max_cost_usd = c.get("max_cost_per_session_usd", 0.10)
        self.max_sub_questions = c.get("max_sub_questions", 5)
        self.episodic_capacity = c.get("episodic_buffer_capacity", 5)
        self.episodic_entry_max = c.get("episodic_entry_max_tokens", 200)
        self.compressed_entry_max = c.get("compressed_entry_max_tokens", 100)
        self.retrieval_top_k = c.get("memory_retrieval_top_k", 8)

        self.system_prompt_budget = 400
        self.query_budget = 200
        self.memory_budget = (
            self.max_context_tokens - self.system_prompt_budget - self.query_budget
        )

        tc = config.get("token_costs", {})
        self.input_cost_per_token = tc.get("input_per_million", 0.14) / 1_000_000
        self.output_cost_per_token = tc.get("output_per_million", 0.28) / 1_000_000

    # ── Cost tracking ───────────────────────────────────────────────

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_cost_per_token +
                output_tokens * self.output_cost_per_token)

    def check_budget(self, session_id: str) -> tuple:
        """Return (remaining_usd, is_within_budget)."""
        spent = self.store.get_session_cost(session_id)
        remaining = self.max_cost_usd - spent
        return remaining, remaining > 0.001  # small epsilon

    def record_cost(self, session_id: str, input_tokens: int,
                    output_tokens: int):
        cost = self.estimate_cost(input_tokens, output_tokens)
        self.store.update_session_cost(session_id, input_tokens,
                                       output_tokens, cost)

    # ── Keyword extraction (lightweight, no extra deps) ─────────────

    _STOP_WORDS = frozenset(
        "the a an is are was were be been being have has had do does did "
        "will would could should may might can shall to of in for on with "
        "at by from as into about between through during before after above "
        "below this that these those it its they them their we our you your "
        "he she his her and but or nor not no so if than too very just also "
        "more most such what which who how when where why all each every "
        "both few many some any other new old like well back much then here "
        "there only over".split()
    )

    def extract_keywords(self, text: str) -> list:
        """Extract top-10 keywords by frequency, excluding stop words."""
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        filtered = [w for w in words if w not in self._STOP_WORDS]
        return [word for word, _ in Counter(filtered).most_common(10)]

    # ── Adding findings ─────────────────────────────────────────────

    def add_finding(self, session_id: str, sub_question: str,
                    finding: str) -> dict:
        """
        Insert a finding into the episodic buffer.  If the buffer overflows,
        the oldest entry is compressed and promoted to Level 2.
        """
        truncated = self.counter.truncate(finding, self.episodic_entry_max)
        token_count = self.counter.count(truncated)
        keywords = self.extract_keywords(sub_question + " " + truncated)

        entry_id = self.store.add_episodic_entry(
            session_id, sub_question, truncated, token_count, keywords
        )

        result = {
            "action": "added_to_episodic",
            "entry_id": entry_id,
            "tokens": token_count,
            "compressed": [],
        }

        # Enforce capacity
        active = self.store.get_active_episodic_entries(session_id)
        while len(active) > self.episodic_capacity:
            oldest = active[0]
            info = self._compress_entry(session_id, oldest)
            result["compressed"].append(info)
            active = self.store.get_active_episodic_entries(session_id)

        return result

    def _compress_entry(self, session_id: str, entry: dict) -> dict:
        """
        Extractive compression: truncate the finding to compressed_entry_max
        tokens and move it to compressed memory.  This avoids an extra LLM
        call (saves cost) at the expense of losing tail-end detail.
        """
        compressed_text = self.counter.truncate(
            entry["finding"], self.compressed_entry_max
        )
        token_count = self.counter.count(compressed_text)

        raw_kw = entry.get("keywords", "[]")
        keywords = json.loads(raw_kw) if isinstance(raw_kw, str) else (raw_kw or [])
        topic = (entry.get("sub_question") or "general")[:100]

        self.store.add_compressed_memory(
            session_id=session_id,
            summary=compressed_text,
            token_count=token_count,
            keywords=keywords,
            topic=topic,
            source_episodic_id=entry["id"],
        )
        self.store.mark_episodic_compressed(entry["id"])

        return {
            "source_id": entry["id"],
            "original_tokens": entry["token_count"],
            "compressed_tokens": token_count,
        }

    # ── Retrieval ───────────────────────────────────────────────────

    def retrieve_context(self, session_id: str, query: str) -> tuple:
        """
        Build a memory context string for the LLM, respecting the memory
        token budget.

        Priority order:
          1. All active episodic entries (newest first — recency bias)
          2. Top-k compressed memories ranked by keyword overlap with query

        Returns (context_string, total_token_count).
        """
        budget = self.memory_budget
        parts: list[str] = []
        total_tokens = 0

        # ── Level 1: episodic buffer ──
        episodic = self.store.get_active_episodic_entries(session_id)
        ep_lines: list[str] = []
        for entry in reversed(episodic):
            line = f"[Finding] Q: {entry['sub_question']}\nA: {entry['finding']}"
            line_tok = self.counter.count(line)
            if total_tokens + line_tok <= budget:
                ep_lines.append(line)
                total_tokens += line_tok
            else:
                remaining = budget - total_tokens
                if remaining > 20:
                    ep_lines.append(self.counter.truncate(line, remaining))
                    total_tokens += remaining
                break

        if ep_lines:
            parts.append("=== Recent Findings ===\n" + "\n---\n".join(ep_lines))

        # ── Level 2: compressed memories (keyword-ranked) ──
        remaining_budget = budget - total_tokens
        if remaining_budget > 50:
            compressed = self.store.get_all_compressed_memories(session_id)
            if compressed:
                qkw = set(self.extract_keywords(query))
                scored = []
                for mem in compressed:
                    raw = mem.get("keywords", "[]")
                    mkw = set(json.loads(raw) if isinstance(raw, str) else (raw or []))
                    scored.append((len(qkw & mkw), mem))
                scored.sort(key=lambda x: x[0], reverse=True)

                cm_lines: list[str] = []
                for _score, mem in scored[: self.retrieval_top_k]:
                    line = f"[Memory] {mem['topic']}: {mem['summary']}"
                    line_tok = self.counter.count(line)
                    if total_tokens + line_tok <= budget:
                        cm_lines.append(line)
                        total_tokens += line_tok
                    else:
                        remaining = budget - total_tokens
                        if remaining > 20:
                            cm_lines.append(self.counter.truncate(line, remaining))
                            total_tokens += remaining
                        break

                if cm_lines:
                    parts.append(
                        "=== Background Knowledge ===\n" + "\n---\n".join(cm_lines)
                    )

        context = "\n\n".join(parts) if parts else "(No prior findings yet)"
        return context, total_tokens

    def get_synthesis_context(self, session_id: str) -> tuple:
        """
        Retrieve ALL accumulated memories for the final synthesis step.
        The synthesis call gets a slightly larger budget (memory_budget + 400)
        because it only happens once per session.
        """
        budget = self.memory_budget + 400
        lines: list[str] = []
        total_tokens = 0

        for mem in self.store.get_all_compressed_memories(session_id):
            line = f"• {mem['topic']}: {mem['summary']}"
            tok = self.counter.count(line)
            if total_tokens + tok <= budget:
                lines.append(line)
                total_tokens += tok

        for entry in self.store.get_active_episodic_entries(session_id):
            line = f"• {entry['sub_question']}: {entry['finding']}"
            tok = self.counter.count(line)
            if total_tokens + tok <= budget:
                lines.append(line)
                total_tokens += tok
            else:
                remaining = budget - total_tokens
                if remaining > 20:
                    lines.append(self.counter.truncate(line, remaining))
                    total_tokens += remaining
                break

        context = "\n".join(lines) if lines else "(No findings available)"
        return context, total_tokens

    # ── Diagnostics ─────────────────────────────────────────────────

    def get_memory_stats(self, session_id: str) -> dict:
        episodic = self.store.get_active_episodic_entries(session_id)
        compressed = self.store.get_all_compressed_memories(session_id)

        ep_tok = sum(e["token_count"] for e in episodic)
        cm_tok = sum(m["token_count"] for m in compressed)
        remaining, within = self.check_budget(session_id)

        return {
            "episodic_entries": len(episodic),
            "episodic_capacity": self.episodic_capacity,
            "episodic_tokens": ep_tok,
            "compressed_entries": len(compressed),
            "compressed_tokens": cm_tok,
            "total_memory_tokens": ep_tok + cm_tok,
            "memory_budget_per_call": self.memory_budget,
            "cost_remaining_usd": round(remaining, 6),
            "within_budget": within,
        }