import json
import numpy as np
import numpy.typing as npt
from typing import Any
from src.parser import FunctionDefinition, ParameterDefinition


class Validator:
    """Validate generated tokens to ensure they form a valid JSON object.

    Maintains the generation context to identify the selected function
    and parameter, allowing token generation to be restricted according
    to the expected parameter type.
    """

    def __init__(self, llm: Any, functions: list[FunctionDefinition]) -> None:
        """Initialize the validator with an LLM and available functions.

        Stores the function definitions and initializes the generation
        context and vocabulary token collections.

        Args:
            llm: Language model instance used to decode generated tokens.
            functions: Available function definitions and their parameters.
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
        """Initialize the vocabulary token collections.

        Iterates through the model vocabulary and classifies tokens
        according to the characters they contain. Creates separate
        collections for structural, numeric and boolean tokens.

        Args:
            vocab_size: Total number of tokens in the model vocabulary.

        Returns:
            None.
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
        """Reset the validator to its initial generation state.

        Clears the generated tokens, decoded buffer and current function
        and parameter references to prevent context from previous
        generations from affecting the new generation.

        Returns:
            The current validator instance with its generation state reset.
        """
        self.buffer = ""
        self.generated = []
        self.current_function = None
        self.current_parameter = None

        return self

    def inside_string(self) -> bool:
        """Check whether the current buffer is inside a JSON string.

        Tracks quotation marks while accounting for escaped characters.

        Returns:
            True if the buffer is currently inside a JSON string,
            False otherwise.
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
        """Retrieve the allowed tokens for the current generation context.

        Selects the vocabulary according to the current JSON state and
        the type of the parameter being generated.

        Returns:
            An array containing the allowed token IDs. Returns None
            if the corresponding vocabulary has not been initialized.
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
        """Advance the generation state with a newly generated token.

        Appends the token ID, updates the decoded buffer and refreshes
        the current function and parameter context.

        Args:
            token_id: ID of the newly generated token.

        Returns:
            The current validator instance with its state updated.
        """
        self.generated.append(token_id)
        self.buffer = self.llm.decode(self.generated)

        self.update_context()

        return self

    def is_accepting(self) -> bool:
        """Check whether the generated buffer contains a valid JSON object.

        Attempts to parse the current buffer and verifies that the
        resulting JSON value is a dictionary.

        Returns:
            True if the buffer contains a valid JSON object,
            False otherwise.
        """
        try:
            result = isinstance(json.loads(self.buffer), dict)
        except json.JSONDecodeError:
            result = False

        return result

    def update_context(self) -> None:
        """Update the current function and parameter generation context.

        Identifies the selected function once its name is completely
        generated. Then identifies the parameter being generated by
        checking its name against the selected function definition.

        Clears the current parameter when its value reaches a comma
        or closing brace, allowing the validator to process the next
        parameter or finish the parameters object.

        Returns:
            None.
        """
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
