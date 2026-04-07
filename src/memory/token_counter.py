import tiktoken


class TokenCounter:
    """
    Accurate token counting using tiktoken with cl100k_base encoding.
    This encoding is used by GPT-4, GPT-3.5-turbo, and is a reasonable
    proxy for DeepSeek and most modern LLMs (within ~5% accuracy).
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        """Count the number of tokens in a string."""
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within max_tokens, decoding back to valid text."""
        if not text:
            return ""
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoding.decode(tokens[:max_tokens])

    def fits(self, text: str, budget: int) -> bool:
        """Check if text fits within a token budget."""
        return self.count(text) <= budget