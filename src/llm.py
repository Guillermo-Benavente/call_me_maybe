import numpy as np
from typing import Any

from src.validator import Validator
from llm_sdk.llm_sdk import Small_LLM_Model


class LLM:

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self._llm = Small_LLM_Model(model_name, device="cpu")
        self.json_grammar = Validator(self)

    def encode(self, text: str) -> Any:
        return self._llm.encode(text)

    def decode(self, ids: Any) -> str:
        return str(self._llm.decode(ids))

    def get_logits(self, input_ids: list[int]) -> list[float]:
        return list(self._llm.get_logits_from_input_ids(input_ids))

    def get_vocab_path(self) -> str:
        return str(self._llm.get_path_to_vocab_file())

    def get_merges_path(self) -> str:
        return str(self._llm.get_path_to_merges_file())

    def get_tokenizer_path(self) -> str:
        return str(self._llm.get_path_to_tokenizer_file())

    def generate_json(self, input_ids: list[int]) -> list[int]:
        generated: list[int] = []
        grammar: Validator = self.json_grammar.initial_state()

        logits = np.asarray(
            self.get_logits(input_ids),
            dtype=np.float32
        )

        if grammar.all_tokens is None:
            grammar.initialize_vocab(len(logits))

        while True:
            allowed: np.ndarray | None = grammar.allowed_tokens()

            if allowed:
                next_id: int = int(allowed[np.argmax(logits[allowed])])
                generated.append(next_id)
                grammar.advance(next_id)

            if grammar.is_accepting:
                break

            logits = np.asarray(
                self.get_logits(input_ids + generated),
                dtype=np.float32
            )

        return generated
