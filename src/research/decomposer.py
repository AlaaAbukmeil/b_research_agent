import json
import re

from src.dify_client import DifyClient
from memory.token_counter import TokenCounter


class QueryDecomposer:
    """
    Takes a complex, multi-part research question and breaks it into
    focused, answerable sub-questions via a Dify workflow.
    """

    def __init__(self, client: DifyClient, max_sub_questions: int = 5):
        self.client = client
        self.max_sub_questions = max_sub_questions
        self.counter = TokenCounter()

    def decompose(self, query: str) -> dict:
        """
        Returns {
            "sub_questions": [str, ...],
            "strategy":      str,
            "input_tokens":  int,
            "output_tokens": int,
        }
        """
        est_input = self.counter.count(query) + 350  # prompt overhead
        try:
            outputs = self.client.run_workflow({"query": query})
            result_text = outputs.get("result", outputs.get("text", ""))
            est_output = self.counter.count(result_text)

            parsed = self._parse(result_text)
            parsed["input_tokens"] = est_input
            parsed["output_tokens"] = est_output
            return parsed

        except Exception as exc:
            print(f"  ⚠ Decomposition failed: {exc}")
            print("  → Falling back to single-question approach")
            return {
                "sub_questions": [query],
                "strategy": "Direct answer (decomposition unavailable)",
                "input_tokens": est_input,
                "output_tokens": 0,
            }

    # ── Internal parsing helpers ────────────────────────────────────

    def _parse(self, text: str) -> dict:
        # Try JSON first
        parsed = self._try_json(text)
        if parsed:
            return parsed

        # Fall back to numbered-list parsing
        return self._parse_numbered(text)

    def _try_json(self, text: str) -> dict | None:
        cleaned = re.sub(r"```json?\s*", "", text)
        cleaned = re.sub(r"```", "", cleaned).strip()
        try:
            data = json.loads(cleaned)
            questions = data.get("sub_questions", data.get("questions", []))
            if isinstance(questions, list) and questions:
                return {
                    "sub_questions": questions[: self.max_sub_questions],
                    "strategy": data.get("strategy", "Systematic decomposition"),
                }
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
        return None

    def _parse_numbered(self, text: str) -> dict:
        questions: list[str] = []
        for line in text.strip().splitlines():
            line = line.strip()
            m = re.match(r"^(?:\d+[\.\)]\s*|[-•]\s+)(.*)", line)
            if m:
                q = m.group(1).strip()
                if len(q) > 10:
                    questions.append(q)

        if not questions:
            questions = [text.strip()[:500]]

        return {
            "sub_questions": questions[: self.max_sub_questions],
            "strategy": "Decomposed from numbered list",
        }