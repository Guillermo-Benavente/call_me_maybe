from typing import Any

from llm_sdk.llm_sdk import Small_LLM_Model


class LLM:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self._llm = Small_LLM_Model(model_name, device="cpu")

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

    def generate(
    self,
    input_ids: list[int],
    max_new_tokens: int = 512,
    eos_ids: tuple[int, ...] = (151645, 151643), # <|im_end|>, <|endoftext|>
    ) -> list[int]:
        ids = list(input_ids)
        generated: list[int] = []
        for _ in range(max_new_tokens):
            logits = self.get_logits(ids)
            next_id = max(range(len(logits)), key=logits.__getitem__)
            ids.append(next_id)
            print(self.decode([next_id]), end="", flush=True)
            generated.append(next_id)
            if next_id in eos_ids:
                break
        return generated