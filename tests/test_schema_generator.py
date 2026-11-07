import json
from pathlib import Path

from src.schema_generator import SchemaConstrainedGenerator
from src.validator import FunctionDefinition, ParameterDefinition
from src.vocabulary import build_byte_to_character_mapping

VOCABULARY_SIZE = 256


class TokenIdentifiers:
    """Minimal stand-in for the tensor returned by the SDK."""

    def __init__(self, identifiers: list[int]) -> None:
        """Store the identifiers returned by the fake encoder."""
        self._identifiers = identifiers

    def __getitem__(self, index: int) -> "TokenIdentifiers":
        """Return the single batch of identifiers."""
        return self

    def tolist(self) -> list[int]:
        """Return the identifiers as a plain Python list."""
        return list(self._identifiers)


class FollowingLanguageModel:
    """Language model that prefers the next byte of a target text.

    Every token is chosen by the model, so a test fails if the generator
    writes a character on its own. The position is derived from the number of
    identifiers appended to the prompt, which also makes repeated logits
    requests at the same position harmless.
    """

    def __init__(self, vocabulary_path: str, target_text: str) -> None:
        """Store the vocabulary path and the text the model must produce."""
        self._vocabulary_path = vocabulary_path
        self._target_bytes = target_text.encode("utf-8")
        self._prompt_token_count = 0

    def encode(self, text: str) -> TokenIdentifiers:
        """Map every UTF-8 byte of the prompt to its single-byte token."""
        encoded_bytes = list(text.encode("utf-8"))
        self._prompt_token_count = len(encoded_bytes)
        return TokenIdentifiers(encoded_bytes)

    def get_logits_from_input_ids(
        self,
        input_token_identifiers: list[int],
    ) -> list[float]:
        """Score the next byte of the target text."""
        logits = [0.0] * VOCABULARY_SIZE
        generated_byte_count = (
            len(input_token_identifiers) - self._prompt_token_count
        )

        if generated_byte_count < len(self._target_bytes):
            logits[self._target_bytes[generated_byte_count]] = 10.0

        return logits

    def get_path_to_vocab_file(self) -> str:
        """Return the path to the generated fake vocabulary file."""
        return self._vocabulary_path


def write_fake_vocabulary(directory: Path) -> str:
    """Write a vocabulary with one token per byte."""
    byte_to_character = build_byte_to_character_mapping()
    raw_vocabulary = {
        byte_to_character[byte_value]: byte_value
        for byte_value in range(256)
    }

    vocabulary_path = directory / "vocabulary.json"
    vocabulary_path.write_text(
        json.dumps(raw_vocabulary),
        encoding="utf-8",
    )

    return str(vocabulary_path)


def build_greet_function() -> FunctionDefinition:
    """Build a function with a single string parameter."""
    return FunctionDefinition(
        name="fn_greet",
        parameters={"name": ParameterDefinition(type="string")},
    )


def build_add_function() -> FunctionDefinition:
    """Build a function with two numeric parameters."""
    return FunctionDefinition(
        name="fn_add",
        parameters={
            "a": ParameterDefinition(type="number"),
            "b": ParameterDefinition(type="number"),
        },
    )


def test_model_emits_every_character_of_a_string_call(
    tmp_path: Path,
) -> None:
    """Keys, separators and braces are emitted by the model too."""
    vocabulary_path = write_fake_vocabulary(tmp_path)
    model = FollowingLanguageModel(
        vocabulary_path,
        '{"name":"fn_greet","parameters":{"name":"Alice"}}',
    )
    generator = SchemaConstrainedGenerator(model)

    generated_text = generator.generate(
        "Greet Alice",
        [build_greet_function()],
    )

    assert generated_text == (
        '{"name":"fn_greet","parameters":{"name":"Alice"}}'
    )


def test_model_emits_numbers_in_schema_order(tmp_path: Path) -> None:
    """Numeric values stop exactly before the following literal."""
    vocabulary_path = write_fake_vocabulary(tmp_path)
    model = FollowingLanguageModel(
        vocabulary_path,
        '{"name":"fn_add","parameters":{"a":2,"b":3}}',
    )
    generator = SchemaConstrainedGenerator(model)

    generated_text = generator.generate(
        "Add 2 and 3",
        [build_add_function()],
    )

    assert generated_text == (
        '{"name":"fn_add","parameters":{"a":2,"b":3}}'
    )


def test_function_name_constraint_allows_only_available_names(
    tmp_path: Path,
) -> None:
    """Only bytes that continue an available name may be selected."""
    vocabulary_path = write_fake_vocabulary(tmp_path)
    generator = SchemaConstrainedGenerator(
        FollowingLanguageModel(vocabulary_path, "")
    )

    allowed_token_identifiers = generator._prefix_token_identifiers(
        "function name",
        (b'fn_add"', b'fn_greet"'),
        b"fn_",
    )

    assert ord("a") in allowed_token_identifiers
    assert ord("g") in allowed_token_identifiers
    assert ord("z") not in allowed_token_identifiers
