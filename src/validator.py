import json
import numpy as np
import numpy.typing as npt
from typing import Any
from src.parser import FunctionDefinition, ParameterDefinition


class Validator:
    """Validate generated tokens to ensure they form a valid JSON object."""

    def __init__(self, llm: Any, functions: list[FunctionDefinition]) -> None:
        """Initialize the validator with an LLM instance.

        Args:
            llm: Language model instance used to decode generated tokens.
        """
        from src.llm import LLM
        self.llm: LLM = llm
        self.functions = {
            function.name: function
            for function in functions
        }

        self.buffer = ""
        self.generated: list[Any] = []

        self.allowed_chars = {
          "structural": set('{}[]":,0123456789.truefalsn- '),
          "number": set('0123456789.-eE,}] '),
          "boolean": set('truefals,}] '),
        }

        self.structural_tokens: npt.NDArray[np.int64] | None = None
        self.number_tokens: npt.NDArray[np.int64] | None = None
        self.boolean_tokens: npt.NDArray[np.int64] | None = None
        self.all_tokens: npt.NDArray[np.int64] | None = None

        self.current_function: FunctionDefinition | None = None
        self.current_parameter: ParameterDefinition | None = None

    def initialize_vocab(self, vocab_size: int) -> None:
        """Initialize the vocabulary token arrays.

        Identifies tokens containing only characters allowed in JSON
        structural elements and stores their IDs separately.

        Args:
            vocab_size: Total number of tokens in the model vocabulary.
        """
        self.all_tokens = np.arange(vocab_size, dtype=np.int64)
        structural = []
        numbers = []
        booleans = []

        for token_id in range(vocab_size):
            text = self.llm.decode([token_id])

            if all(char in self.allowed_chars["structural"] for char in text):
                structural.append(token_id)

            if all(char in self.allowed_chars["number"] for char in text):
                numbers.append(token_id)

            if all(char in self.allowed_chars["boolean"] for char in text):
                booleans.append(token_id)

        self.structural_tokens = np.array(structural, dtype=np.int64)
        self.number_tokens = np.array(numbers, dtype=np.int64)
        self.boolean_tokens = np.array(booleans, dtype=np.int64)

    def initial_state(self) -> "Validator":
        """Reset the validator state.

        Clears the generated tokens and the decoded text buffer.

        Returns:
            The current validator instance.
        """
        self.buffer = ""
        self.generated = []

        return self

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

    def allowed_tokens(self) -> npt.NDArray[np.int64] | None:
        """Return the tokens allowed in the current generation state.

        All vocabulary tokens are allowed inside JSON strings.
        Outside strings, only structural tokens are allowed.

        Returns:
            An array of allowed token IDs, or None if not initialized.
        """
        if self.inside_string():
            return self.all_tokens

        if self.current_parameter is not None:
            parameter_type = self.current_parameter.type

            if parameter_type in ("integer", "float", "number"):
                return self.number_tokens

            if parameter_type == "boolean":
                return self.boolean_tokens

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

        self.update_context()

        return self

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

    def update_context(self) -> None:
        if self.current_parameter is not None:
            if self.buffer.endswith((",", "}")):
                self.current_parameter = None
                return
        if self.current_function is None:
            if '"name": "' in self.buffer:
                function_name = self.buffer.split('"name": "')[1].split('"')[0]

                if '"name": "' + function_name + '"' in self.buffer:
                    self.current_function = self.functions.get(function_name)

        if (
            self.current_function is not None
            and self.current_parameter is None
        ):
            if '"parameters": {' in self.buffer:
                parameters = self.buffer.split('"parameters": {')[1]

                if '"' in parameters:
                    parameter_name = parameters.split('"')[1]

                    if (
                        parameter_name
                        in self.current_function.parameters.keys()
                        and f'"{parameter_name}"' in parameters
                    ):
                        self.current_parameter = (
                            self.current_function.parameters
                                .get(parameter_name)
                        )
