import json
from pathlib import Path

from src.vocabulary import Vocabulary


def test_vocabulary_recovers_token_bytes(tmp_path: Path) -> None:
    """The byte-level vocabulary must recover the original token bytes."""
    vocabulary_path = tmp_path / "vocabulary.json"
    vocabulary_path.write_text(
        json.dumps({
            "name": 0,
            "Ġname": 1,
            "!": 2,
        }),
        encoding="utf-8",
    )

    vocabulary = Vocabulary.from_vocabulary_file(str(vocabulary_path))

    assert vocabulary.token_bytes(0) == b"name"
    assert vocabulary.token_bytes(1) == b" name"
    assert vocabulary.token_bytes(2) == b"!"
    assert vocabulary.token_bytes(3) is None
