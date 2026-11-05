from typing import Any, Callable

from llm_sdk import Small_LLM_Model

from .json_state_machine import JsonGenerationState, JsonStateMachine
from .parser import ResponseParser
from .prompt_builder import PromptBuilder
from .validator import FunctionDefinition
from .vocabulary import Vocabulary


class FunctionSelector:
    """Selects functions using guided JSON token generation."""

    def __init__(
        self,
        model: Small_LLM_Model,
        max_new_tokens: int = 64,
    ) -> None:
        """Initialize the selector and load the token vocabulary once.

        Args:
            model: Language model wrapper provided by the SDK.
            max_new_tokens: Maximum number of tokens to generate.
        """
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.vocabulary = Vocabulary.from_vocabulary_file(
            model.get_path_to_vocab_file()
        )
        self._allowed_token_identifiers_by_state: dict[
            tuple[JsonGenerationState, int],
            set[int],
        ] = {}
        self._state_predicates: dict[
            JsonGenerationState,
            Callable[[bytes], bool],
        ] = {
            JsonGenerationState.OBJECT_START: self._starts_json_object,
            JsonGenerationState.KEY_START: self._starts_json_key,
            JsonGenerationState.COLON: self._starts_colon,
            JsonGenerationState.VALUE_END: self._starts_value_end,
        }

    def select(
        self,
        user_prompt: str,
        functions: list[FunctionDefinition],
    ) -> dict[str, Any]:
        """Generate the function call for a natural-language request.

        Args:
            user_prompt: Request written in natural language.
            functions: Available function definitions.

        Returns:
            The parsed function call produced by the model.

        Raises:
            ValueError: If generation does not produce valid JSON.
        """
        prompt = PromptBuilder.build(
            user_prompt=user_prompt,
            functions=functions,
        )
        response = self._generate(prompt)
        return ResponseParser.parse(response)

    def _generate(self, prompt: str) -> str:
        """Generate a JSON object token by token.

        Args:
            prompt: Complete prompt sent to the language model.

        Returns:
            The generated JSON text.

        Raises:
            ValueError: If the model cannot complete the JSON object.
        """
        input_token_identifiers = self.model.encode(prompt)[0].tolist()
        generated_token_bytes = bytearray()
        state_machine = JsonStateMachine()

        print("\nGenerating response...", flush=True)

        for _ in range(self.max_new_tokens):
            logits = self.model.get_logits_from_input_ids(
                input_token_identifiers
            )
            self._filter_logits(logits, state_machine.state)

            next_token_identifier = max(
                range(len(logits)),
                key=lambda token_identifier: logits[token_identifier],
            )

            if logits[next_token_identifier] == float("-inf"):
                raise ValueError(
                    "No valid token is available in state "
                    f"{state_machine.state.name}."
                )

            token_bytes = self.vocabulary.token_bytes(
                next_token_identifier
            )
            if token_bytes is None:
                raise ValueError(
                    "The model selected an unknown token identifier: "
                    f"{next_token_identifier}."
                )

            input_token_identifiers.append(next_token_identifier)
            consumed_byte_count = state_machine.advance(token_bytes)
            generated_token_bytes.extend(
                token_bytes[:consumed_byte_count]
            )

            print(
                token_bytes[:consumed_byte_count].decode(
                    "utf-8",
                    errors="replace",
                ),
                end="",
                flush=True,
            )

            if state_machine.is_complete:
                print("\nJSON generation stopped.", flush=True)

                try:
                    return generated_token_bytes.decode("utf-8")
                except UnicodeDecodeError as error:
                    raise ValueError(
                        "The generated JSON is not valid UTF-8."
                    ) from error

        raise ValueError("The model did not complete the JSON object.")

    def _filter_logits(
        self,
        logits: list[float],
        state: JsonGenerationState,
    ) -> None:
        """Mask every token that is not valid for the current state.

        Args:
            logits: Raw score for every token in the vocabulary.
            state: Current syntactic state of the JSON object.
        """
        allowed_token_identifiers = self._allowed_token_identifiers(
            state,
            len(logits),
        )

        for token_identifier in range(len(logits)):
            token_bytes = self.vocabulary.token_bytes(token_identifier)

            if token_bytes is None:
                logits[token_identifier] = float("-inf")
            elif (
                allowed_token_identifiers is not None
                and token_identifier not in allowed_token_identifiers
            ):
                logits[token_identifier] = float("-inf")

    def _allowed_token_identifiers(
        self,
        state: JsonGenerationState,
        vocabulary_size: int,
    ) -> set[int] | None:
        """Return the token identifiers allowed in a given state.

        Args:
            state: Current syntactic state of the JSON object.
            vocabulary_size: Number of logits returned by the model.

        Returns:
            The allowed identifiers, or ``None`` when the state is free.
        """
        predicate = self._state_predicates.get(state)

        if predicate is None:
            return None

        cache_key = (state, vocabulary_size)
        allowed_token_identifiers = (
            self._allowed_token_identifiers_by_state.get(cache_key)
        )

        if allowed_token_identifiers is None:
            allowed_token_identifiers = {
                token_identifier
                for token_identifier in range(vocabulary_size)
                if (
                    token_bytes := self.vocabulary.token_bytes(
                        token_identifier
                    )
                )
                is not None
                and predicate(token_bytes)
            }
            self._allowed_token_identifiers_by_state[
                cache_key
            ] = allowed_token_identifiers

        return allowed_token_identifiers

    @staticmethod
    def _starts_json_object(token_bytes: bytes) -> bool:
        """Return whether a token starts a JSON object."""
        return token_bytes.startswith(b"{")

    @staticmethod
    def _starts_json_key(token_bytes: bytes) -> bool:
        """Return whether a token starts a JSON object key."""
        return token_bytes.startswith(b'"')

    @staticmethod
    def _starts_colon(token_bytes: bytes) -> bool:
        """Return whether a token starts a JSON key-value separator."""
        return token_bytes.startswith(b":")

    @staticmethod
    def _starts_value_end(token_bytes: bytes) -> bool:
        """Return whether a token starts an argument separator or close."""
        return token_bytes.startswith(b",") or token_bytes.startswith(b"}")
