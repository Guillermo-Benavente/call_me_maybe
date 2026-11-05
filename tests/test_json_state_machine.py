from src.json_state_machine import JsonStateMachine


def test_state_machine_stops_at_the_end_of_the_object() -> None:
    """Bytes after the closing brace must not be consumed."""
    state_machine = JsonStateMachine()

    consumed_bytes = state_machine.advance(b'{"a": 1}trailing')

    assert consumed_bytes == 8
    assert state_machine.is_complete is True


def test_state_machine_tracks_nested_objects() -> None:
    """A nested object must not complete the outer object."""
    state_machine = JsonStateMachine()

    consumed_bytes = state_machine.advance(b'{"a": {"b": 1}}rest')

    assert consumed_bytes == 15
    assert state_machine.is_complete is True


def test_state_machine_ignores_braces_inside_strings() -> None:
    """Closing braces inside a string must not complete the object."""
    state_machine = JsonStateMachine()

    consumed_bytes = state_machine.advance(b'{"a": "}"}')

    assert consumed_bytes == 10
    assert state_machine.is_complete is True
