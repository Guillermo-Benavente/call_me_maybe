from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ParameterDefinition(BaseModel):
    """Definition of a single function parameter."""

    model_config = ConfigDict(extra="allow")

    type: str


class FunctionDefinition(BaseModel):
    """Definition of a function that the language model can call."""

    model_config = ConfigDict(extra="allow")

    name: str
    description: str = ""
    parameters: dict[str, ParameterDefinition] = Field(default_factory=dict)
    returns: ParameterDefinition | None = None


class FunctionCallResult(BaseModel):
    """Function call produced by the language model."""

    name: str
    parameters: dict[str, Any]


def value_matches_type(value: Any, expected_type: str) -> bool:
    """Check whether a generated value matches a definition type.

    Args:
        value: Value produced by the language model.
        expected_type: Type name declared in the function definition.

    Returns:
        ``True`` when the value matches the declared type.
    """
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)

    if expected_type == "string":
        return isinstance(value, str)

    if expected_type == "boolean":
        return isinstance(value, bool)

    if expected_type == "object":
        return isinstance(value, dict)

    if expected_type == "array":
        return isinstance(value, list)

    if expected_type == "null":
        return value is None

    return False


class FunctionValidator:
    """Validates function definitions and generated function calls."""

    @staticmethod
    def parse_function_definitions(raw_data: Any) -> list[FunctionDefinition]:
        """Validate raw function definitions and convert them to models.

        Args:
            raw_data: Parsed JSON content from the definitions file.

        Returns:
            The validated function definitions.

        Raises:
            ValueError: If the data does not contain valid definitions.
        """
        if isinstance(raw_data, dict):
            raw_data = raw_data.get("functions", [])

        if not isinstance(raw_data, list):
            raise ValueError("Function definitions must be a JSON array.")

        try:
            return [
                FunctionDefinition.model_validate(definition)
                for definition in raw_data
            ]
        except ValidationError as error:
            raise ValueError(
                f"Invalid function definition: {error}"
            ) from error

    @staticmethod
    def validate_function_call(
        result: dict[str, Any],
        functions: list[FunctionDefinition],
    ) -> bool:
        """Check that a generated call matches the available definitions.

        Args:
            result: Parsed function call produced by the language model.
            functions: Available function definitions.

        Returns:
            ``True`` when the call is valid for the given definitions.
        """
        try:
            function_call = FunctionCallResult.model_validate(result)
        except ValidationError:
            return False

        functions_by_name = {
            function.name: function for function in functions
        }
        function = functions_by_name.get(function_call.name)

        if function is None:
            return False

        if set(function_call.parameters) != set(function.parameters):
            return False

        parameter_definitions = function.parameters.items()

        for parameter_name, parameter_definition in parameter_definitions:
            parameter_value = function_call.parameters[parameter_name]

            if not value_matches_type(
                parameter_value,
                parameter_definition.type,
            ):
                return False

        return True

    @staticmethod
    def build_output(prompt: str, result: dict[str, Any]) -> dict[str, Any]:
        """Build one entry with exactly the keys required by the subject.

        Args:
            prompt: Original natural-language request.
            result: Parsed function call produced by the language model.

        Returns:
            An object containing ``prompt``, ``name`` and ``parameters``.
        """
        try:
            function_call = FunctionCallResult.model_validate(result)
        except ValidationError:
            return {
                "prompt": prompt,
                "name": "",
                "parameters": {},
            }

        return {
            "prompt": prompt,
            "name": function_call.name,
            "parameters": function_call.parameters,
        }
