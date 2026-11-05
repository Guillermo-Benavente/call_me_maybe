import json
from typing import Optional


def build_byte_to_character_mapping() -> dict[int, str]:
    """Build the byte-to-character mapping used by byte-level BPE.

    The vocabulary file stores tokens as text encoded with the GPT-2
    byte-to-unicode mapping. This function builds the forward mapping so
    the original bytes of every token can be recovered.

    Returns:
        A dictionary that maps every byte value to its encoded character.
    """
    printable_bytes = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    byte_values = printable_bytes[:]
    character_codes = printable_bytes[:]

    next_character_code = 0
    for byte_value in range(256):
        if byte_value not in byte_values:
            byte_values.append(byte_value)
            character_codes.append(256 + next_character_code)
            next_character_code += 1

    return {
        byte_value: chr(character_code)
        for byte_value, character_code in zip(byte_values, character_codes)
    }


_BYTE_TO_CHARACTER = build_byte_to_character_mapping()
_CHARACTER_TO_BYTE = {
    character: byte_value
    for byte_value, character in _BYTE_TO_CHARACTER.items()
}


class Vocabulary:
    """Maps token identifiers to their original byte sequences."""

    def __init__(
        self,
        token_bytes_by_identifier: list[Optional[bytes]],
    ) -> None:
        """Store the token byte table built from the vocabulary file."""
        self._token_bytes_by_identifier = token_bytes_by_identifier

    @classmethod
    def from_vocabulary_file(cls, vocabulary_file_path: str) -> "Vocabulary":
        """Build a vocabulary from the JSON file returned by the SDK.

        Args:
            vocabulary_file_path: Path to the tokenizer vocabulary file.

        Returns:
            A vocabulary that can translate token identifiers to bytes.

        Raises:
            OSError: If the vocabulary file cannot be read.
            ValueError: If the vocabulary file does not contain valid data.
        """
        with open(vocabulary_file_path, "r", encoding="utf-8") as file:
            raw_vocabulary: dict[str, int] = json.load(file)

        if not raw_vocabulary:
            raise ValueError("The vocabulary file is empty.")

        highest_token_identifier = max(raw_vocabulary.values())
        token_bytes_by_identifier: list[Optional[bytes]] = [
            None
        ] * (highest_token_identifier + 1)

        for encoded_token, token_identifier in raw_vocabulary.items():
            try:
                token_bytes = bytes(
                    _CHARACTER_TO_BYTE[character]
                    for character in encoded_token
                )
            except KeyError:
                token_bytes = encoded_token.encode("utf-8")

            token_bytes_by_identifier[token_identifier] = token_bytes

        return cls(token_bytes_by_identifier)

    def token_bytes(self, token_identifier: int) -> Optional[bytes]:
        """Return the raw bytes of a token identifier.

        Args:
            token_identifier: Numeric identifier of the token.

        Returns:
            The token bytes, or ``None`` when the identifier is unknown.
        """
        if 0 <= token_identifier < len(self._token_bytes_by_identifier):
            return self._token_bytes_by_identifier[token_identifier]
        return None

    @property
    def size(self) -> int:
        """Return the number of identifiers stored in the vocabulary."""
        return len(self._token_bytes_by_identifier)
