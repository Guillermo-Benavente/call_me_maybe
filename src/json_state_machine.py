from enum import Enum, auto


class JsonGenerationState(Enum):
    """Syntactic states used while a JSON object is generated."""

    OBJECT_START = auto()
    KEY_START = auto()
    KEY_CONTENT = auto()
    COLON = auto()
    VALUE = auto()
    STRING_VALUE = auto()
    COMPOUND_VALUE = auto()
    VALUE_END = auto()
    COMPLETE = auto()


class JsonStateMachine:
    """Tracks the JSON structure while tokens are appended to the output."""

    def __init__(self) -> None:
        """Initialize the machine at the beginning of a JSON object."""
        self.state = JsonGenerationState.OBJECT_START
        self.nesting_depth = 0
        self.inside_string = False
        self.escape_pending = False

    @property
    def is_complete(self) -> bool:
        """Return whether the outer JSON object has been closed."""
        return self.state is JsonGenerationState.COMPLETE

    def advance(self, token_bytes: bytes) -> int:
        """Advance the state machine with the bytes of a generated token.

        Args:
            token_bytes: Raw bytes produced by the language model.

        Returns:
            The number of bytes consumed before the JSON object closed. When
            the object is still open, every byte is consumed.
        """
        for byte_index, byte_value in enumerate(token_bytes):
            character = chr(byte_value)

            if self.inside_string:
                if self.escape_pending:
                    self.escape_pending = False
                elif character == "\\":
                    self.escape_pending = True
                elif character == '"':
                    self.inside_string = False

                    if self.state is JsonGenerationState.KEY_CONTENT:
                        self.state = JsonGenerationState.COLON
                    elif self.state is JsonGenerationState.STRING_VALUE:
                        self.state = JsonGenerationState.VALUE_END
                continue

            if character == '"':
                self.inside_string = True

                if self.state is JsonGenerationState.KEY_START:
                    self.state = JsonGenerationState.KEY_CONTENT
                elif self.state is JsonGenerationState.VALUE:
                    self.state = JsonGenerationState.STRING_VALUE

            elif character == "{":
                self.nesting_depth += 1
                self.state = JsonGenerationState.KEY_START

            elif character == "[":
                self.nesting_depth += 1
                self.state = JsonGenerationState.VALUE

            elif character in ("}", "]"):
                self.nesting_depth -= 1

                if self.nesting_depth == 0:
                    self.state = JsonGenerationState.COMPLETE
                    return byte_index + 1

                self.state = JsonGenerationState.VALUE_END

            elif character == ":":
                self.state = JsonGenerationState.VALUE

            elif character == ",":
                self.state = (
                    JsonGenerationState.KEY_START
                    if self.nesting_depth == 1
                    else JsonGenerationState.VALUE
                )

            elif character in (" ", "\n", "\r", "\t"):
                continue

            elif self.state is JsonGenerationState.VALUE:
                self.state = JsonGenerationState.VALUE_END

        return len(token_bytes)
