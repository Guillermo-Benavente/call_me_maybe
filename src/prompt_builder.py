import json
from typing import Any


class PromptBuilder:
    """Build prompts for function calling."""

    @staticmethod
    def build(
        user_prompt: str,
        functions_json: str,
    ) -> str:
        """Build the complete model prompt."""

        return f"""
You are a function calling assistant.

Your task is to select the most appropriate function
to fulfill the user's request.

You MUST follow the provided function definitions.
Return only valid JSON without explanations or markdown.

FUNCTION DEFINITIONS:
{functions_json}

USER REQUEST:
{user_prompt}

JSON RESPONSE:
"""