from src.validator import (
    FunctionDefinition,
    FunctionValidator,
    ParameterDefinition,
)


def build_add_function() -> FunctionDefinition:
    """Build the addition function used by the validator tests."""
    return FunctionDefinition(
        name="fn_add_numbers",
        description="Add two numbers.",
        parameters={
            "a": ParameterDefinition(type="number"),
            "b": ParameterDefinition(type="number"),
        },
    )


def test_build_output_uses_the_required_keys() -> None:
    """The output entry must contain prompt, name and parameters."""
    result = FunctionValidator.build_output(
        "What is the sum of 2 and 3?",
        {
            "name": "fn_add_numbers",
            "parameters": {"a": 2, "b": 3},
        },
    )

    assert result == {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {"a": 2, "b": 3},
    }


def test_build_output_returns_empty_values_for_invalid_data() -> None:
    """Invalid model data must not break the output contract."""
    result = FunctionValidator.build_output("prompt", {})

    assert result == {
        "prompt": "prompt",
        "name": "",
        "parameters": {},
    }


def test_validate_function_call_accepts_matching_types() -> None:
    """A call with every required parameter and correct types is valid."""
    functions = [build_add_function()]

    result = FunctionValidator.validate_function_call(
        {
            "name": "fn_add_numbers",
            "parameters": {"a": 2, "b": 3},
        },
        functions,
    )

    assert result is True


def test_validate_function_call_rejects_missing_parameters() -> None:
    """All parameters declared in the definition are required."""
    functions = [build_add_function()]

    result = FunctionValidator.validate_function_call(
        {
            "name": "fn_add_numbers",
            "parameters": {"a": 2},
        },
        functions,
    )

    assert result is False


def test_validate_function_call_rejects_wrong_types() -> None:
    """Parameter values must match their declared types."""
    functions = [build_add_function()]

    result = FunctionValidator.validate_function_call(
        {
            "name": "fn_add_numbers",
            "parameters": {"a": "two", "b": 3},
        },
        functions,
    )

    assert result is False


def test_validate_function_call_rejects_unknown_functions() -> None:
    """The selected function must exist in the definitions file."""
    functions = [build_add_function()]

    result = FunctionValidator.validate_function_call(
        {
            "name": "fn_unknown",
            "parameters": {},
        },
        functions,
    )

    assert result is False
