import json
import numpy as np
from typing import Any


class Validator:
    """Validate generated tokens to ensure they form a valid JSON object."""

    def __init__(self, llm: Any) -> None:
        """Initialize the validator with an LLM instance.

        Args:
            llm: Language model instance used to decode generated tokens.
        """
        from src.llm import LLM
        self.llm: LLM = llm
        self.buffer = ""
        self.generated: list[Any] = []

        self.allowed_chars = set('{}[]":,0123456789.truefalsn- ')
        self.structural_tokens: np.ndarray | None = None
        self.all_tokens: np.ndarray | None = None

    def initialize_vocab(self, vocab_size: int) -> None:
        """Initialize the vocabulary token arrays.

        Identifies tokens containing only characters allowed in JSON
        structural elements and stores their IDs separately.

        Args:
            vocab_size: Total number of tokens in the model vocabulary.
        """
        self.all_tokens = np.arange(vocab_size, dtype=np.int64)
        structural = []

        for token_id in range(vocab_size):
            text = self.llm.decode([token_id])

            if text and all(char in self.allowed_chars for char in text):
                structural.append(token_id)

        self.structural_tokens = np.array(
            structural,
            dtype=np.int64
        )

    def initial_state(self) -> "Validator":
        """Reset the validator state.

        Clears the generated tokens and the decoded text buffer.

        Returns:
            The current validator instance.
        """
        self.buffer = ""
        self.generated = []

        return self

    @property
    def inside_string(self) -> bool:
        """Check whether the current buffer is inside a JSON string.

        Tracks quotation marks while accounting for escaped characters.

        Returns:
            True if the current buffer is inside a string, False otherwise.
        """
        escaped = False
        inside = False

        for char in self.buffer:
            if escaped:
                escaped = False
                continue

            if char == "\\":
                escaped = True
                continue

            if char == '"':
                inside = not inside

        return inside

    def allowed_tokens(self) -> np.ndarray | None:
        """Return the tokens allowed in the current generation state.

        All vocabulary tokens are allowed inside JSON strings.
        Outside strings, only structural tokens are allowed.

        Returns:
            An array of allowed token IDs, or None if not initialized.
        """
        if self.inside_string:
            return self.all_tokens

        return self.structural_tokens

    def advance(self, token_id: Any) -> "Validator":
        """Add a generated token and update the decoded buffer.

        Args:
            token_id: ID of the newly generated token.

        Returns:
            The current validator instance.
        """
        self.generated.append(token_id)
        self.buffer = self.llm.decode(self.generated)

        return self

    @property
    def is_accepting(self) -> bool:
        """Check whether the current buffer contains a valid JSON object.

        Returns:
            True if the buffer can be parsed as a dictionary,
            False otherwise.
        """
        try:
            result = isinstance(json.loads(self.buffer), dict)
        except json.JSONDecodeError:
            result = False

        return result
