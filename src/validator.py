
import json
import numpy as np
from typing import Any


class Validator:

    def __init__(self, llm: Any) -> None:
        from src.llm import LLM
        self.llm: LLM = llm
        self.buffer = ""
        self.generated: list[Any] = []

        self.allowed_chars = set('{}[]":,0123456789.truefalsn- ')
        self.structural_tokens: np.ndarray | None = None
        self.all_tokens: np.ndarray | None = None

    def initialize_vocab(self, vocab_size: int) -> None:
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
        self.buffer = ""
        self.generated = []

        return self

    @property
    def inside_string(self) -> bool:
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
        if self.inside_string:
            return self.all_tokens

        return self.structural_tokens

    def advance(self, token_id: Any) -> "Validator":
        self.generated.append(token_id)
        self.buffer = self.llm.decode(self.generated)

        return self

    @property
    def is_accepting(self) -> bool:
        try:
            result = isinstance(json.loads(self.buffer), dict)
        except json.JSONDecodeError:
            result = False

        return result
