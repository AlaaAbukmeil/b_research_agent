from src.dify_client import DifyClient
from ..memory.manager import MemoryManager
from ..memory.token_counter import TokenCounter


class Synthesizer:
    """
    Final stage: reads ALL accumulated memory (compressed + episodic)
    and produces a single, coherent answer to the original research query.
    """

    def __init__(self, client: DifyClient, memory_manager: MemoryManager):
        self.client = client
        self.memory = memory_manager
        self.counter = TokenCounter()

    def synthesize(self, session_id: str, original_query: str) -> dict:
        """
        Returns {
            "answer":         str,
            "context_tokens": int,
            "input_tokens":   int,
            "output_tokens":  int,
        }
        """
        context, ctx_tokens = self.memory.get_synthesis_context(session_id)
        est_input = (
            self.counter.count(original_query)
            + self.counter.count(context)
            + self.memory.system_prompt_budget
        )

        try:
            outputs = self.client.run_workflow({
                "original_query": original_query,
                "research_findings": context,
            })
            answer = outputs.get("result", outputs.get("text",
                                 "No synthesis result"))
            if not isinstance(answer, str):
                answer = str(answer)
            est_output = self.counter.count(answer)

            self.memory.record_cost(session_id, est_input, est_output)

            return {
                "answer": answer,
                "context_tokens": ctx_tokens,
                "input_tokens": est_input,
                "output_tokens": est_output,
            }

        except Exception as exc:
            print(f"  ⚠ Synthesis failed: {exc}")
            return {
                "answer": f"(Synthesis failed — raw findings below)\n\n{context}",
                "context_tokens": ctx_tokens,
                "input_tokens": est_input,
                "output_tokens": 0,
            }