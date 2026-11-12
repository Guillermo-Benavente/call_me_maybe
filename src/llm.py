import numpy as np
from typing import Any

from src.validator import Validator
from llm_sdk.llm_sdk import Small_LLM_Model


class LLM:
    """Provide an interface for language model operations
    and JSON generation."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """Initialize the language model and its JSON grammar validator.

        Loads the specified model on the CPU and creates a validator
        to constrain generated tokens to valid JSON structures.

        Args:
            model_name: Name or path of the language model to load.
                Defaults to "Qwen/Qwen3-0.6B".
        """
        self._llm = Small_LLM_Model(model_name, device="cpu")
        self.json_grammar = Validator(self)

    def encode(self, text: str) -> Any:
        """Encode a text string into token IDs.

        Args:
            text: Text to encode.

        Returns:
            Token IDs representing the input text.
        """
        return self._llm.encode(text)

    def decode(self, ids: Any) -> str:
        """Decode token IDs into a text string.

        Args:
            ids: Token IDs to decode.

        Returns:
            Decoded text representation of the token IDs.
        """
        return str(self._llm.decode(ids))

    def get_logits(self, input_ids: list[int]) -> list[float]:
        """Retrieve the model logits for the given input tokens.

        Args:
            input_ids: Token IDs provided as model input.

        Returns:
            A list of logits representing the model's prediction
            scores for the next token.
        """
        return list(self._llm.get_logits_from_input_ids(input_ids))

    def get_vocab_path(self) -> str:
        """Get the path to the model vocabulary file.

        Returns:
            Path to the vocabulary file as a string.
        """
        return str(self._llm.get_path_to_vocab_file())

    def get_merges_path(self) -> str:
        """Get the path to the model merges file.

        Returns:
            Path to the merges file as a string.
        """
        return str(self._llm.get_path_to_merges_file())

    def get_tokenizer_path(self) -> str:
        """Get the path to the model tokenizer file.

        Returns:
            Path to the tokenizer file as a string.
        """
        return str(self._llm.get_path_to_tokenizer_file())

    def generate_json(self, input_ids: list[int]) -> list[int]:
        """Generate tokens constrained to form a valid JSON object.

        Uses the validator to restrict token selection according to
        the current JSON generation state. At each iteration, selects
        the allowed token with the highest model logit until a valid
        JSON object is generated.

        Args:
            input_ids: Initial token IDs used as model input.

        Returns:
            A list of generated token IDs forming a valid JSON object.

        Raises:
            RuntimeError: If the grammar validator has no allowed tokens
                during generation.
        """
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

            if allowed is None or allowed.size == 0:
                raise RuntimeError("Grammar has no allowed tokens")

            if allowed is not None and allowed.size > 0:
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
