from src.dify_client import DifyClient
from ..memory.manager import MemoryManager
from ..memory.token_counter import TokenCounter


class ResearchExecutor:
    """
    Researches a single sub-question. Before calling the LLM it retrieves
    relevant memory context (within the per-call token budget) so the model
    can build on prior findings rather than starting from scratch.
    """

    def __init__(self, client: DifyClient, memory_manager: MemoryManager):
        self.client = client
        self.memory = memory_manager
        self.counter = TokenCounter()

    def research(self, session_id: str, sub_question: str,
                 original_query: str) -> dict:
        """
        Returns {
            "finding":              str,
            "memory_context_tokens": int,
            "input_tokens":         int,
            "output_tokens":        int,
            "memory_update":        dict,
            "budget_exceeded":      bool,
        }
        """
        remaining, within = self.memory.check_budget(session_id)
        if not within:
            return {
                "finding": "(Budget exceeded — skipped)",
                "memory_context_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "memory_update": {"action": "budget_exceeded"},
                "budget_exceeded": True,
            }

        context, ctx_tokens = self.memory.retrieve_context(
            session_id, sub_question
        )
        est_input = (
            self.counter.count(sub_question)
            + self.counter.count(context)
            + self.memory.system_prompt_budget
        )

        try:
            outputs = self.client.run_workflow({
                "sub_question": sub_question,
                "original_query": original_query,
                "memory_context": context,
            })

            finding = outputs.get("result", outputs.get("text",
                                  "No result returned"))
            if not isinstance(finding, str):
                finding = str(finding)
            est_output = self.counter.count(finding)

            self.memory.record_cost(session_id, est_input, est_output)
            mem_update = self.memory.add_finding(
                session_id, sub_question, finding
            )

            return {
                "finding": finding,
                "memory_context_tokens": ctx_tokens,
                "input_tokens": est_input,
                "output_tokens": est_output,
                "memory_update": mem_update,
                "budget_exceeded": False,
            }

        except Exception as exc:
            print(f"  ⚠ Research failed for: {sub_question[:60]}… — {exc}")
            self.memory.record_cost(session_id, est_input, 0)
            return {
                "finding": f"(Research failed: {str(exc)[:100]})",
                "memory_context_tokens": ctx_tokens,
                "input_tokens": est_input,
                "output_tokens": 0,
                "memory_update": {"action": "error"},
                "budget_exceeded": False,
            }