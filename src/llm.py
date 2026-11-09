import numpy as np
import time
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
        return self._llm.decode(ids)

    def get_logits(self, input_ids: list[int]) -> list[float]:
        return self._llm.get_logits_from_input_ids(input_ids)

    def get_vocab_path(self) -> str:
        return self._llm.get_path_to_vocab_file()

    def get_merges_path(self) -> str:
        return self._llm.get_path_to_merges_file()

    def get_tokenizer_path(self) -> str:
        return self._llm.get_path_to_tokenizer_file()

    def generate_json(self, input_ids: list[int]) -> list[int]:

        generated = []
        grammar = self.json_grammar.initial_state()

        timings = {
            "get_logits": 0.0,
            "initialize_vocab": 0.0,
            "allowed_tokens": 0.0,
            "selection": 0.0,
            "advance": 0.0,
            "is_accepting": 0.0
        }

        iterations = 0
        total_start = time.perf_counter()

        start = time.perf_counter()
        logits = np.asarray(
            self.get_logits(input_ids),
            dtype=np.float32
        )
        timings["get_logits"] += time.perf_counter() - start

        # Inicialización del vocabulario
        if grammar.all_tokens is None:
            start = time.perf_counter()
            grammar.initialize_vocab(len(logits))
            timings["initialize_vocab"] += time.perf_counter() - start

        while True:
            start = time.perf_counter()
            allowed = grammar.allowed_tokens()
            timings["allowed_tokens"] += time.perf_counter() - start

            start = time.perf_counter()
            next_id = int(allowed[np.argmax(logits[allowed])])
            print(self.decode([next_id]), end="", flush=True)
            timings["selection"] += time.perf_counter() - start

            generated.append(next_id)

            start = time.perf_counter()
            grammar.advance(next_id)
            timings["advance"] += time.perf_counter() - start

            start = time.perf_counter()
            accepting = grammar.is_accepting
            timings["is_accepting"] += time.perf_counter() - start

            iterations += 1

            if accepting:
                break

            start = time.perf_counter()
            logits = np.asarray(
                self.get_logits(input_ids + generated),
                dtype=np.float32
            )
            timings["get_logits"] += time.perf_counter() - start

        total = time.perf_counter() - total_start

        print("\n========== GENERATION PROFILE ==========")
        print(f"Total: {total:.4f}s")
        print(f"Generated tokens: {iterations}")

        print("\n--- LLM ---")
        for name, value in timings.items():
            print(f"{name:20}: {value:.6f}s ({value * 1000:.3f} ms)")

        print("\n--- VALIDATOR INTERNAL ---")
        for name, value in grammar.timings.items():
            print(f"{name:20}: {value:.6f}s ({value * 1000:.3f} ms)")

        print("========================================\n")

        return generated
