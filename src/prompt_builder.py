import json

from .validator import FunctionDefinition


class PromptBuilder:
    """Builds prompts for language model function calling."""

    @staticmethod
    def build(
        user_prompt: str,
        functions: list[FunctionDefinition],
    ) -> str:
        """Build the prompt that describes the available functions.

        The prompt is kept compact on purpose: a small model follows short,
        single-line instructions more reliably than an indented copy of the
        definitions, and the return type is not needed to build the call.

        Args:
            user_prompt: Natural-language request from the user.
            functions: Available function definitions.

        Returns:
            The complete prompt sent to the language model.
        """
        functions_json = json.dumps(
            [
                function.model_dump(
                    exclude={"returns"},
                    exclude_none=True,
                )
                for function in functions
            ],
            ensure_ascii=False,
        )

        return f"""{PromptBuilder._INSTRUCTIONS}

Available functions:
{functions_json}

User request:
{user_prompt}

Response:"""

    _INSTRUCTIONS = "\n".join([
        "Choose one of the available functions for the user's request.",
        "Extract its arguments from the user's request. "
        "Do not execute the function.",
        'Return only one JSON object with the keys "name" and '
        '"parameters", in that order.',
        "Write compact JSON on a single line, without spaces outside "
        "string values.",
        "Escape special characters as required by JSON.",
        "Do not add extra keys, explanations or Markdown.",
        "Use the exact function name and parameter names, and respect the "
        "declared parameter types.",
        "Never write JSON braces inside an argument value.",
    ])
