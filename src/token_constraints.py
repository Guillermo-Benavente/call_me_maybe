"""Token-level constraints used while a JSON value is generated."""

from dataclasses import dataclass
from enum import Enum, auto

HEXADECIMAL_DIGIT_BYTES = frozenset(b"0123456789abcdefABCDEF")
ESCAPE_BYTES = frozenset(b'"\\/bfnrtu')

ZERO_BYTE = ord("0")
NINE_BYTE = ord("9")
MINUS_SIGN_BYTE = ord("-")
DECIMAL_POINT_BYTE = ord(".")
LOWERCASE_EXPONENT_BYTE = ord("e")
UPPERCASE_EXPONENT_BYTE = ord("E")
PLUS_SIGN_BYTE = ord("+")
QUOTE_BYTE = ord('"')
BACKSLASH_BYTE = ord("\\")
LOWERCASE_U_BYTE = ord("u")
LOWEST_CONTROL_BYTE = 0x20


class NumberState(Enum):
    """Grammar states of a JSON number being generated."""

    START = auto()
    MINUS_SIGN = auto()
    ZERO = auto()
    INTEGER_DIGITS = auto()
    DECIMAL_POINT = auto()
    FRACTION_DIGITS = auto()
    EXPONENT_MARK = auto()
    EXPONENT_SIGN = auto()
    EXPONENT_DIGITS = auto()


COMPLETE_NUMBER_STATES = frozenset({
    NumberState.ZERO,
    NumberState.INTEGER_DIGITS,
    NumberState.FRACTION_DIGITS,
    NumberState.EXPONENT_DIGITS,
})


def _is_digit(byte_value: int) -> bool:
    """Return whether a byte value is an ASCII decimal digit."""
    return ZERO_BYTE <= byte_value <= NINE_BYTE


def advance_number_state(
    state: NumberState,
    byte_value: int,
    allow_fraction_and_exponent: bool,
) -> NumberState | None:
    """Advance the JSON number grammar by one byte.

    Args:
        state: Current grammar state.
        byte_value: Byte that will be appended to the number.
        allow_fraction_and_exponent: Whether decimal and exponent parts are
            allowed. Integer parameters use ``False``.

    Returns:
        The next state, or ``None`` when the byte is not valid.
    """
    if state is NumberState.START:
        if byte_value == MINUS_SIGN_BYTE:
            return NumberState.MINUS_SIGN
        if byte_value == ZERO_BYTE:
            return NumberState.ZERO
        if _is_digit(byte_value):
            return NumberState.INTEGER_DIGITS
        return None

    if state is NumberState.MINUS_SIGN:
        if byte_value == ZERO_BYTE:
            return NumberState.ZERO
        if _is_digit(byte_value):
            return NumberState.INTEGER_DIGITS
        return None

    if state is NumberState.ZERO:
        if allow_fraction_and_exponent and byte_value == DECIMAL_POINT_BYTE:
            return NumberState.DECIMAL_POINT
        if allow_fraction_and_exponent and byte_value in (
            LOWERCASE_EXPONENT_BYTE,
            UPPERCASE_EXPONENT_BYTE,
        ):
            return NumberState.EXPONENT_MARK
        return None

    if state is NumberState.INTEGER_DIGITS:
        if _is_digit(byte_value):
            return NumberState.INTEGER_DIGITS
        if allow_fraction_and_exponent and byte_value == DECIMAL_POINT_BYTE:
            return NumberState.DECIMAL_POINT
        if allow_fraction_and_exponent and byte_value in (
            LOWERCASE_EXPONENT_BYTE,
            UPPERCASE_EXPONENT_BYTE,
        ):
            return NumberState.EXPONENT_MARK
        return None

    if state is NumberState.DECIMAL_POINT:
        if _is_digit(byte_value):
            return NumberState.FRACTION_DIGITS
        return None

    if state is NumberState.FRACTION_DIGITS:
        if _is_digit(byte_value):
            return NumberState.FRACTION_DIGITS
        if byte_value in (
            LOWERCASE_EXPONENT_BYTE,
            UPPERCASE_EXPONENT_BYTE,
        ):
            return NumberState.EXPONENT_MARK
        return None

    if state is NumberState.EXPONENT_MARK:
        if byte_value in (PLUS_SIGN_BYTE, MINUS_SIGN_BYTE):
            return NumberState.EXPONENT_SIGN
        if _is_digit(byte_value):
            return NumberState.EXPONENT_DIGITS
        return None

    if state is NumberState.EXPONENT_SIGN:
        if _is_digit(byte_value):
            return NumberState.EXPONENT_DIGITS
        return None

    if state is NumberState.EXPONENT_DIGITS:
        if _is_digit(byte_value):
            return NumberState.EXPONENT_DIGITS
        return None

    return None


def advance_number_bytes(
    state: NumberState,
    token_bytes: bytes,
    allow_fraction_and_exponent: bool,
) -> NumberState | None:
    """Advance the JSON number grammar through a whole token.

    Args:
        state: Current grammar state.
        token_bytes: Bytes of the candidate token.
        allow_fraction_and_exponent: Whether decimal and exponent parts are
            allowed.

    Returns:
        The resulting state, or ``None`` when the token is not valid.
    """
    current_state: NumberState | None = state

    for byte_value in token_bytes:
        if current_state is None:
            return None

        current_state = advance_number_state(
            current_state,
            byte_value,
            allow_fraction_and_exponent,
        )

    return current_state


@dataclass(frozen=True)
class StringState:
    """Grammar state of a JSON string body being generated."""

    escape_pending: bool = False
    hexadecimal_digits_remaining: int = 0
    closed: bool = False


def advance_string_state(
    state: StringState,
    token_bytes: bytes,
) -> StringState | None:
    """Advance the JSON string grammar through a whole token.

    The closing quote must be the last byte of the token, so the generated
    text never continues after the string has ended.

    Args:
        state: Current grammar state.
        token_bytes: Bytes of the candidate token.

    Returns:
        The resulting state, or ``None`` when the token is not valid.
    """
    if state.closed:
        return None

    escape_pending = state.escape_pending
    hexadecimal_digits_remaining = state.hexadecimal_digits_remaining
    last_byte_index = len(token_bytes) - 1

    for byte_index, byte_value in enumerate(token_bytes):
        if hexadecimal_digits_remaining > 0:
            if byte_value not in HEXADECIMAL_DIGIT_BYTES:
                return None

            hexadecimal_digits_remaining -= 1
            continue

        if escape_pending:
            if byte_value not in ESCAPE_BYTES:
                return None

            escape_pending = False

            if byte_value == LOWERCASE_U_BYTE:
                hexadecimal_digits_remaining = 4
            continue

        if byte_value == BACKSLASH_BYTE:
            escape_pending = True
            continue

        if byte_value == QUOTE_BYTE:
            if byte_index != last_byte_index:
                return None

            return StringState(closed=True)

        if byte_value < LOWEST_CONTROL_BYTE:
            return None

    return StringState(
        escape_pending=escape_pending,
        hexadecimal_digits_remaining=hexadecimal_digits_remaining,
    )
