class PromptBuilder:
    """Build prompts for function calling."""

    @staticmethod
    def build(
        user_prompt: str,
        functions_json: str,
    ) -> str:
        """Build a complete prompt with function definitions and user request.

        Constructs the instructions that guide the model to select exactly
        one available function and return its arguments in JSON format.

        Args:
            user_prompt: User's request to be processed by the model.
            functions_json: JSON string containing the available function
                definitions and their parameters.

        Returns:
            A formatted prompt containing the system instructions,
            function definitions, user request, and required
            JSON output format.
        """
        return (
            "You are a function calling assistant.\n"
            "Choose exactly one function from the list and fill its "
            "parameters.\n Answer with ONE JSON object and nothing else.\n"
            "No markdown, no code fences, no explanations, no extra text.\n\n"
            f"Functions: {functions_json}\n\n"
            f"User request: {user_prompt}\n\n"
            "Required output format:\n"
            '{"name": "<function_name>", "parameters": {<arguments>}}\n\n'
            "JSON:\n"
        )
