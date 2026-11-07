from src.token_constraints import (
    COMPLETE_NUMBER_STATES,
    NumberState,
    StringState,
    advance_number_bytes,
    advance_string_state,
)


def number_state_after(
    text: str,
    allow_fraction_and_exponent: bool = True,
) -> NumberState | None:
    """Return the grammar state after reading a whole text as a number."""
    return advance_number_bytes(
        NumberState.START,
        text.encode("ascii"),
        allow_fraction_and_exponent,
    )


def test_number_accepts_complete_values() -> None:
    """Complete JSON numbers must reach a complete grammar state."""
    assert number_state_after("0") is NumberState.ZERO
    assert number_state_after("42") is NumberState.INTEGER_DIGITS
    assert number_state_after("-7") is NumberState.INTEGER_DIGITS
    assert number_state_after("3.5") is NumberState.FRACTION_DIGITS
    assert number_state_after("1e10") is NumberState.EXPONENT_DIGITS
    assert number_state_after("1E-4") is NumberState.EXPONENT_DIGITS


def test_number_accepts_incomplete_prefixes() -> None:
    """A value that is still being written must remain a valid prefix."""
    assert number_state_after("") is NumberState.START
    assert number_state_after("-") is NumberState.MINUS_SIGN
    assert number_state_after("1.") is NumberState.DECIMAL_POINT
    assert number_state_after("1e") is NumberState.EXPONENT_MARK
    assert number_state_after("1e+") is NumberState.EXPONENT_SIGN


def test_number_rejects_invalid_forms() -> None:
    """Invalid JSON numbers must be rejected."""
    assert number_state_after("01") is None
    assert number_state_after(".5") is None
    assert number_state_after("2,3") is None
    assert number_state_after("--1") is None


def test_incomplete_number_is_not_complete() -> None:
    """Only complete grammar states may end a numeric value."""
    assert number_state_after("12") in COMPLETE_NUMBER_STATES
    assert number_state_after("1.") not in COMPLETE_NUMBER_STATES
    assert number_state_after("1e") not in COMPLETE_NUMBER_STATES
    assert number_state_after("") not in COMPLETE_NUMBER_STATES


def test_integer_rejects_fraction_and_exponent() -> None:
    """Integer parameters must not accept decimal or exponent parts."""
    allow = False

    assert number_state_after("42", allow) is NumberState.INTEGER_DIGITS
    assert number_state_after("4.2", allow) is None
    assert number_state_after("4e2", allow) is None


def test_string_closes_only_at_the_end_of_a_token() -> None:
    """A closing quote followed by more bytes must be rejected."""
    open_state = StringState()

    assert advance_string_state(open_state, b"hello") == StringState()
    assert advance_string_state(open_state, b'hello"') == StringState(
        closed=True
    )
    assert advance_string_state(open_state, b'hel"lo') is None
    assert advance_string_state(StringState(closed=True), b"x") is None


def test_string_tracks_escapes() -> None:
    """Escapes must be complete and must use valid JSON escape characters."""
    assert advance_string_state(StringState(), b"\\") == StringState(
        escape_pending=True
    )
    assert advance_string_state(StringState(), b'\\"') == StringState()
    assert advance_string_state(StringState(), b"\\q") is None
    assert advance_string_state(StringState(), b"\\u00e9") == StringState()
    assert advance_string_state(StringState(), b"\\u00e") == StringState(
        hexadecimal_digits_remaining=1
    )


def test_string_rejects_control_characters() -> None:
    """Raw control characters are not allowed inside a JSON string."""
    assert advance_string_state(StringState(), b"\x01") is None
    assert advance_string_state(StringState(), b"\n") is None
