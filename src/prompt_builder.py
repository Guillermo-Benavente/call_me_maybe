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

        Args:
            user_prompt: Natural-language request from the user.
            functions: Available function definitions.

        Returns:
            The complete prompt sent to the language model.
        """
        functions_json = json.dumps(
            [
                function.model_dump(exclude_none=True)
                for function in functions
            ],
            indent=2,
            ensure_ascii=False,
        )

        return f"""
You are an assistant that selects functions based on user requests.

Available functions:
{functions_json}

Instructions:
- Select the most appropriate function for the user's request.
- Use only the functions provided above.
- Extract the required arguments from the user's request.
- Do not invent arguments or function names.
- Do not explain your reasoning.
- Do not provide analysis or intermediate steps.
- Do not include greetings or additional text.
- Return exactly one JSON object.
- Do not use Markdown or code fences.

Expected output format:
{{
    "name": "function_name",
    "parameters": {{
        "parameter": "value"
    }}
}}

User request:
{user_prompt}
""".strip()
